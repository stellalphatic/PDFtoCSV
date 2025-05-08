document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded and parsed");
    
    // Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const extractBtn = document.getElementById('extract-btn');
    const uploadForm = document.getElementById('upload-form');
    
    console.log("Upload form found:", uploadForm !== null);
    console.log("Upload area found:", uploadArea !== null);
    console.log("File input found:", fileInput !== null);

    if (!uploadArea || !fileInput || !fileName || !extractBtn || !uploadForm) {
        console.error("One or more required elements not found");
        return;
    }

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);

    // Handle file selection via input
    fileInput.addEventListener('change', handleFiles, false);

    // Handle click on upload area - trigger file input
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    // Form submission
    uploadForm.addEventListener('submit', function(e) {
        console.log("Form submit event triggered");
        // Don't disable the button here, let the form submission happen
        extractBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Processing...';
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        uploadArea.classList.add('highlight');
    }

    function unhighlight() {
        uploadArea.classList.remove('highlight');
    }

    function handleDrop(e) {
        console.log("File dropped");
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            handleFiles({ target: { files: files } });
        }
    }

    function handleFiles(e) {
        console.log("File selected");
        const files = e.target.files;
        if (files.length === 0) {
            return;
        }
        
        const file = files[0];
        console.log("File type:", file.type);
        
        // Check if file is a PDF
        if (file.type !== 'application/pdf') {
            fileName.textContent = 'Error: Please upload a PDF file';
            fileName.classList.add('text-danger');
            fileName.classList.remove('text-success');
            extractBtn.disabled = true;
            return;
        }
        
        // Update UI with file name
        fileName.textContent = file.name;
        fileName.classList.remove('text-danger');
        fileName.classList.add('text-success');
        
        // Enable extract button
        extractBtn.disabled = false;
        console.log("Extract button enabled");
    }

    // Initialize feature animations
    initFeatureAnimations();
    
    function initFeatureAnimations() {
        console.log("Initializing feature animations");
        const features = document.querySelectorAll('.feature-item');
        console.log("Feature items found:", features.length);
        
        if (features.length === 0) {
            return;
        }
        
        // Add initial animation class to first set of features
        document.querySelectorAll('.feature-item').forEach((feature, index) => {
            setTimeout(() => {
                feature.classList.add('feature-animate');
            }, index * 100); // Stagger the animations
        });
    }
});
