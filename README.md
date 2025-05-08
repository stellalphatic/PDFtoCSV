# Advanced PDF Invoice Table Extractor

A comprehensive tool for accurately extracting invoice tables from PDF files, handling various formats, multi-page tables, and filtering out noise.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Web Application](#web-application)
   - [Command Line Tools](#command-line-tools)
6. [Extraction Technologies](#extraction-technologies)
7. [File Descriptions](#file-descriptions)
8. [Extraction Methods](#extraction-methods)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

## Overview

This application is designed to extract invoice tables from PDF files and convert them to CSV format. It uses multiple advanced techniques including OCR, computer vision, and AI-enhanced layout analysis to handle a wide variety of invoice formats, including those with complex layouts, multi-page tables, and varying structures.

The system employs a cascade of extraction methods, starting with the most advanced AI-based techniques and falling back to simpler methods if needed. This approach ensures the highest possible success rate across different invoice types.

## Features

- **Multi-Method Extraction**: Uses multiple extraction techniques with automatic fallback
- **AI-Enhanced Analysis**: Utilizes computer vision and machine learning for better table detection
- **Handles Complex Layouts**: Works with tables without clear boundaries or grid lines
- **Multi-Page Support**: Correctly handles tables that span multiple pages
- **Noise Filtering**: Intelligently identifies and extracts only the relevant invoice table
- **Web Interface**: Easy-to-use web application for uploading and processing invoices
- **Command-Line Tools**: Scriptable tools for batch processing
- **Format Normalization**: Standardizes extracted data for easier use

## Requirements

- Python 3.7+
- Tesseract OCR
- Poppler
- OpenCV
- Flask (for web interface)
- Additional Python libraries (listed in requirements.txt)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/stellalphatic/invoice-table-extractor.git
   cd invoice-table-extractor
   ```

2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install system dependencies:
   
   - For Debian/Ubuntu:
     ```
     sudo apt-get update
     sudo apt-get install -y tesseract-ocr poppler-utils
     ```
   
   - For macOS:
     ```
     brew install tesseract poppler
     ```
   
   - For Windows:
     Download and install:
     - [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
     - [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)

## Usage

### Web Application

The web application provides an easy-to-use interface for uploading and processing invoice PDFs.

1. Start the Flask application:
   ```
   python main.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload a PDF invoice using the web interface and download the resulting CSV file.

### Command Line Tools

Three different extraction tools are provided, each with its own strengths:

#### 1. Basic Extractor

```
python invoice_extractor.py <pdf_path> [--output <output_path>] [--debug]
```

Example:
```
python invoice_extractor.py invoices/sample.pdf --output extracted_table.csv
```

#### 2. Advanced Extractor

```
python invoice_table_extractor_advanced.py <pdf_path> [-o <output_path>] [-d]
```

Example:
```
python invoice_table_extractor_advanced.py invoices/complex_invoice.pdf -o result.csv -d
```

#### 3. AI-Enhanced Extractor

```
python ai_invoice_extractor.py <pdf_path> [-o <output_path>] [-d]
```

Example:
```
python ai_invoice_extractor.py invoices/difficult_invoice.pdf -o ai_result.csv -d
```

## Extraction Technologies

The system uses a combination of technologies to achieve accurate table extraction:

1. **PDF Parsing**: Using `pdfplumber` to extract text and basic table structures
2. **OCR**: Using `Tesseract OCR` via `pytesseract` for processing scanned or image-based PDFs
3. **Computer Vision**: Using `OpenCV` for detecting lines, table structures, and layout analysis
4. **Image Processing**: Advanced preprocessing techniques to improve OCR accuracy
5. **Text Analysis**: Pattern matching and positional analysis for detecting table structures from raw text
6. **AI-Enhanced Layout Analysis**: Machine learning-based approaches to detect and analyze document structure

## File Descriptions

| File | Description |
|------|-------------|
| `main.py` | Flask web application for the user interface |
| `invoice_extractor.py` | Basic invoice table extractor with multiple extraction methods |
| `invoice_table_extractor_advanced.py` | Advanced extractor with specialized invoice table detection |
| `ai_invoice_extractor.py` | AI-enhanced extractor using computer vision and advanced techniques |
| `table_detector.py` | Helper module for detecting table structures in images |
| `table_extraction_enhancement.py` | Module with methods to improve extraction quality |
| `enhanced_invoice_processor.py` | Processing pipeline for enhanced extraction |
| `utils.py` | Utility functions used across the application |

## Extraction Methods

The system employs several extraction methods, applied in sequence from most to least sophisticated:

### 1. AI-Enhanced Computer Vision Method

- Uses image processing to detect table structures
- Identifies rows and columns based on lines and text alignment
- Maps extracted text to the detected table structure
- Applies invoice-specific heuristics to improve accuracy

Parameters:
- `confidence_threshold`: Minimum confidence for detection (default: 0.7)
- Debug mode for visualizing detected table regions

### 2. Layout Analysis Method

- Analyzes text positions and alignments to detect table structures without visible grid lines
- Works well for tables with consistent text alignment but no visible borders
- Uses intelligent column mapping based on position

Parameters:
- Text position tolerance for grouping rows and columns

### 3. Structured Extraction Method

- Uses `pdfplumber`'s built-in table extraction capabilities
- Tries multiple table settings to optimize extraction
- Best for PDFs with well-defined table structures

Parameters:
- Multiple variations of vertical and horizontal detection strategies

### 4. OCR-Based Method

- Converts PDF pages to images and applies OCR
- Applies sophisticated preprocessing to improve OCR quality
- Uses position information to reconstruct table structure

Parameters:
- OCR configuration options
- Image preprocessing parameters

### 5. Text Pattern Analysis Method

- Analyzes raw text output to identify table structures
- Looks for patterns in lines, spaces, and formatting
- Uses header detection and row alignment analysis

Parameters:
- Pattern matching settings for different table styles

## Troubleshooting

If extraction results are not as expected, try the following:

1. **Try Different Extractors**: Different invoice formats work better with different extraction methods
2. **Enable Debug Mode**: Use the `-d` or `--debug` flag to get more information
3. **Check PDF Quality**: Better quality PDFs generally produce better results
4. **Check System Dependencies**: Ensure Tesseract OCR and Poppler are correctly installed
5. **Pre-process Difficult PDFs**: For scanned or low-quality PDFs, try improving the scan quality

## License

This project is licensed under the MIT License - see the LICENSE file for details.
