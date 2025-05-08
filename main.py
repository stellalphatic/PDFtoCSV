import os
import sys
import logging
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

# Import all extractors
from invoice_extractor import InvoiceTableExtractor
from invoice_table_extractor_advanced import AdvancedInvoiceExtractor
from ai_invoice_extractor import AIInvoiceExtractor

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_key")

# Configure upload folder
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'invoice_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OUTPUT_FOLDER = os.path.join(tempfile.gettempdir(), 'invoice_outputs')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize the invoice extractors
# Set paths to external tools if needed
tesseract_path = os.environ.get('TESSERACT_PATH', None)
poppler_path = os.environ.get('POPPLER_PATH', None)

# Original extractor as final fallback
original_extractor = InvoiceTableExtractor(
    tesseract_path=tesseract_path,
    poppler_path=poppler_path,
    debug=True
)

# Enhanced extractor as secondary option
advanced_extractor = AdvancedInvoiceExtractor(
    tesseract_path=tesseract_path,
    debug=True
)

# AI-enhanced extractor as primary option
ai_extractor = AIInvoiceExtractor(
    tesseract_path=tesseract_path,
    debug=True
)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    logger.debug("Upload file called")
    logger.debug(f"Request files: {request.files}")
    
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    logger.debug(f"File received: {file.filename}")
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the uploaded file
        filename = secure_filename(file.filename if file.filename is not None else "uploaded_file.pdf")
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.debug(f"Saving file to: {pdf_path}")
        file.save(pdf_path)
        
        # Generate output filename
        output_name = os.path.splitext(filename)[0] + '.csv'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)
        logger.debug(f"Output will be saved to: {output_path}")
        
        try:
            # First try with AI-enhanced extractor
            logger.info(f"Starting AI-enhanced invoice extraction for: {filename}")
            results = ai_extractor.extract_from_pdf(pdf_path, output_path)
            
            # If AI extractor fails, try advanced extractor as first fallback
            if results is None or results.empty:
                logger.info(f"AI extraction failed, trying advanced extractor")
                results = advanced_extractor.extract_from_pdf(pdf_path, output_path)
                
                # If advanced extractor also fails, try original extractor as final fallback
                if results is None or results.empty:
                    logger.info(f"Advanced extraction failed, trying original extractor")
                    results = original_extractor.extract_from_pdf(pdf_path, output_path)
            
            logger.info(f"Extraction complete for: {filename}")
            
            if results is None or results.empty:
                flash('No valid table found in the invoice', 'warning')
                return redirect(url_for('index'))
            
            # Return the CSV file for download
            logger.info(f"Sending CSV file: {output_path}")
            return send_file(
                output_path,
                mimetype='text/csv',
                download_name=output_name,
                as_attachment=True
            )
        
        except Exception as e:
            logger.error(f"Error processing invoice: {str(e)}")
            flash(f'Error processing invoice: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload a PDF file.', 'error')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """Show information about the application."""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
