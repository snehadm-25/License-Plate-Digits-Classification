const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const predictBtn = document.getElementById('predict-btn');
const removeBtn = document.getElementById('remove-btn');
const previewContainer = document.getElementById('preview-container');
const dropZoneContent = document.querySelector('.drop-zone-content');
const imagePreview = document.getElementById('image-preview');
const resultContainer = document.getElementById('result-container');
const loader = document.getElementById('loader');
const predictionValue = document.getElementById('prediction-value');
const confidenceValue = document.getElementById('confidence-value');

let selectedFile = null;

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
});

// Click to Browse
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.classList.remove('hidden');
        dropZoneContent.classList.add('hidden');
        predictBtn.disabled = false;
        resultContainer.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// Remove Image
removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    previewContainer.classList.add('hidden');
    dropZoneContent.classList.remove('hidden');
    predictBtn.disabled = true;
    resultContainer.classList.add('hidden');
});

// Predict
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    // UI State
    predictBtn.disabled = true;
    loader.classList.remove('hidden');
    resultContainer.classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        
        predictionValue.textContent = data.prediction;
        confidenceValue.textContent = data.confidence;
        
        resultContainer.classList.remove('hidden');
    } catch (error) {
        console.error(error);
        alert('An error occurred while classifying the image.');
    } finally {
        loader.classList.add('hidden');
        predictBtn.disabled = false;
    }
});
