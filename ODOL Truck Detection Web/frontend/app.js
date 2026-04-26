const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('imageUpload');
const fileName = document.getElementById('fileName');

// Show file name when selected
fileInput.addEventListener('change', function () {
    if (fileInput.files[0]) {
        fileName.textContent = fileInput.files[0].name;
    }
});

// Drag & drop events
dropZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', function () {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', function (e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    fileInput.files = e.dataTransfer.files;
    fileName.textContent = e.dataTransfer.files[0].name;
});

// Detect button
document.getElementById('detectBtn').addEventListener('click', async function () {
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image first!');
        return;
    }

    const preview = document.getElementById('preview');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';

    document.getElementById('result').innerText = 'Processing...';

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();

    if (data.detected) {
        document.getElementById('result').innerText =
            data.label + ' (' + (data.confidence * 100).toFixed(1) + '%)';
        preview.src = 'data:image/jpeg;base64,' + data.annotated_image;
    } else {
        document.getElementById('result').innerText = 'No truck detected! Please try again with a different image.';
    }
});
