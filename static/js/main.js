// JazzVibe-AI Main JavaScript
// Handles UI interactions and music generation API calls

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const numNotesSlider = document.getElementById('num-notes');
    const notesValue = document.getElementById('notes-value');
    const temperatureSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('temp-value');
    const generateBtn = document.getElementById('generate-btn');
    const loadingContainer = document.getElementById('loading-container');
    const resultPanel = document.getElementById('result-panel');
    const progressBar = document.getElementById('progress-bar');
    const loadingStatus = document.getElementById('loading-status');
    const downloadBtn = document.getElementById('download-btn');
    const regenerateBtn = document.getElementById('regenerate-btn');
    const notesCount = document.getElementById('notes-count');

    // Update slider values
    numNotesSlider.addEventListener('input', function() {
        notesValue.textContent = this.value;
    });

    temperatureSlider.addEventListener('input', function() {
        tempValue.textContent = parseFloat(this.value).toFixed(1);
    });

    // Generate button click handler
    generateBtn.addEventListener('click', function() {
        generateMusic();
    });

    // Regenerate button click handler
    regenerateBtn.addEventListener('click', function() {
        generateMusic();
    });

    // Generate music function
    async function generateMusic() {
        // Get parameters
        const numNotes = parseInt(numNotesSlider.value);
        const temperature = parseFloat(temperatureSlider.value);

        // Show loading state
        showLoading();
        generateBtn.disabled = true;

        try {
            // Simulate progress
            await simulateProgress();

            // Call API
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    num_notes: numNotes,
                    temperature: temperature
                })
            });

            const data = await response.json();

            if (data.success) {
                // Show result
                showResult(data.filename);
            } else {
                throw new Error(data.error || 'Generation failed');
            }

        } catch (error) {
            console.error('Error generating music:', error);
            alert('Failed to generate music. Please try again.');
            hideLoading();
            generateBtn.disabled = false;
        }
    }

    // Simulate progress animation
    async function simulateProgress() {
        const statuses = [
            'Initializing AI model...',
            'Analyzing jazz patterns...',
            'Generating melody...',
            'Adding harmonies...',
            'Finalizing composition...'
        ];

        for (let i = 0; i <= 100; i += 10) {
            await new Promise(resolve => setTimeout(resolve, 200));
            progressBar.style.width = i + '%';
            
            const statusIndex = Math.floor(i / 25);
            if (statuses[statusIndex]) {
                loadingStatus.textContent = statuses[statusIndex];
            }
        }
    }

    // Show loading state
    function showLoading() {
        loadingContainer.classList.remove('hidden');
        resultPanel.classList.add('hidden');
        progressBar.style.width = '0%';
        loadingStatus.textContent = 'Initializing...';
    }

    // Hide loading state
    function hideLoading() {
        loadingContainer.classList.add('hidden');
    }

    // Show result
    function showResult(filename) {
        hideLoading();
        resultPanel.classList.remove('hidden');
        generateBtn.disabled = false;

        // Set download link
        downloadBtn.href = `/download/${filename}`;
        
        // Update notes count display
        notesCount.textContent = numNotesSlider.value;
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Spacebar to generate (if not focused on inputs)
        if (e.code === 'Space' && !e.target.matches('input')) {
            e.preventDefault();
            if (!generateBtn.disabled) {
                generateMusic();
            }
        }
    });
});
