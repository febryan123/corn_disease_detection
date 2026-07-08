document.addEventListener('DOMContentLoaded', () => {
    const uploadContainer = document.getElementById('uploadContainer');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const uploadContent = document.getElementById('uploadContent');
    const previewContent = document.getElementById('previewContent');
    const imagePreview = document.getElementById('imagePreview');
    const btnRemove = document.getElementById('btnRemove');
    const btnAnalyze = document.getElementById('btnAnalyze');
    
    const loadingState = document.getElementById('loadingState');
    const resultSection = document.getElementById('resultSection');
    
    // Result elements
    const predictedClass = document.getElementById('predictedClass');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceCircle = document.getElementById('confidenceCircle');
    const recommendationText = document.getElementById('recommendationText');
    const probList = document.getElementById('probList');
    const resultCard = document.querySelector('.result-card');

    let currentFile = null;

    // --- Upload Logic ---
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    
    uploadContainer.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, () => {
            uploadContainer.classList.add('drag-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadContainer.addEventListener(eventName, () => {
            uploadContainer.classList.remove('drag-active');
        }, false);
    });

    uploadContainer.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        if (files.length) {
            handleFile(files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert("Harap unggah file gambar (JPG/PNG).");
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadContent.classList.add('hidden');
            previewContent.classList.remove('hidden');
            btnAnalyze.disabled = false;
            
            // Hide result section if it was previously visible
            resultSection.classList.add('hidden');
        };
        
        reader.readAsDataURL(file);
    }

    // Remove image
    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation(); // prevent triggering upload container click
        resetUpload();
    });

    function resetUpload() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        uploadContent.classList.remove('hidden');
        previewContent.classList.add('hidden');
        btnAnalyze.disabled = true;
        resultSection.classList.add('hidden');
    }

    // --- Analyze Logic ---
    btnAnalyze.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Updates for loading
        btnAnalyze.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                displayResult(result);
            } else {
                alert("Terjadi kesalahan: " + result.error);
            }
        } catch (error) {
            console.error("Error:", error);
            alert("Terjadi kesalahan koneksi ke server.");
        } finally {
            // Restore UI
            loadingState.classList.add('hidden');
            btnAnalyze.classList.remove('hidden');
        }
    });

    function displayResult(data) {
        // Update texts
        predictedClass.textContent = data.prediction;
        confidenceText.textContent = data.confidence;
        recommendationText.innerHTML = data.recommendation;

        // Clean previous status classes
        resultCard.classList.remove('result-success', 'result-warning');
        
        // Apply new status class based on backend decision
        if (data.status_color === "success") {
            resultCard.classList.add('result-success');
        } else {
            resultCard.classList.add('result-warning');
        }

        // Animate circular chart
        const confValue = parseFloat(data.confidence);
        // stroke-dasharray values are percentage, remainder
        setTimeout(() => {
            confidenceCircle.setAttribute('stroke-dasharray', `${confValue}, 100`);
        }, 100);

        // Update probability list
        probList.innerHTML = '';
        
        const sortedProbabilities = Object.entries(data.all_probabilities).sort((a, b) => {
            return parseFloat(b[1]) - parseFloat(a[1]);
        });

        for (const [className, prob] of sortedProbabilities) {
            const li = document.createElement('li');
            li.className = 'prob-item';
            
            const probFloat = parseFloat(prob);
            let barColor = 'rgba(0, 230, 118, 0.15)';
            
            // Highlight the predicted class
            if (className === data.prediction) {
                li.style.border = '1px solid var(--primary-color)';
                barColor = 'rgba(0, 230, 118, 0.3)';
            }
            
            li.innerHTML = `
                <div class="prob-bar" style="width: 0%; background: ${barColor};"></div>
                <div class="prob-content">
                    <span class="prob-name">${className}</span>
                    <span class="prob-val">${prob}</span>
                </div>
            `;
            probList.appendChild(li);
            
            // Animate progress bar
            setTimeout(() => {
                li.querySelector('.prob-bar').style.width = `${probFloat}%`;
            }, 50);
        }

        // Show result section
        resultSection.classList.remove('hidden');
        
        // Scroll to result smoothly
        setTimeout(() => {
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
});
