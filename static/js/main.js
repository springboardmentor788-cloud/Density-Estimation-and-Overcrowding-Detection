document.addEventListener('DOMContentLoaded', () => {
    const thresholdVal = document.getElementById('threshold-val');
    const modeVal = document.getElementById('mode-val');
    const alertBanner = document.getElementById('alert-banner');

    const sourceInput = document.getElementById('source-input');
    const loadBtn = document.getElementById('load-btn');
    const systemStatus = document.getElementById('status-dot');

    const fileUpload = document.getElementById('file-upload');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadStatus = document.getElementById('upload-status');

    uploadBtn.addEventListener('click', () => fileUpload.click());

    fileUpload.addEventListener('change', async () => {
        if (!fileUpload.files.length) return;
        
        const file = fileUpload.files[0];
        console.log('File selected:', file.name, 'Size:', file.size);
        const formData = new FormData();
        formData.append('file', file);

        try {
            uploadStatus.textContent = 'Uploading ' + file.name + '...';
            uploadBtn.style.borderColor = 'var(--primary)';
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            console.log('Upload response status:', response.status);
            const data = await response.json();
            console.log('Upload response data:', data);
            
            if (data.status === 'success') {
                uploadStatus.textContent = 'Ready: ' + file.name;
                uploadBtn.style.borderColor = '#4ade80';
                
                // Refresh video feed
                const videoFeed = document.querySelector('.video-feed');
                if (videoFeed) {
                    videoFeed.src = videoFeed.src.split('?')[0] + '?t=' + new Date().getTime();
                }
            } else {
                uploadStatus.textContent = 'Upload failed: ' + data.message;
                console.error('Upload error:', data.message);
            }
        } catch (error) {
            console.error('Upload error:', error);
            uploadStatus.textContent = 'Server Error: ' + error.message;
        }
    });

    loadBtn.addEventListener('click', async () => {
        const source = sourceInput.value.trim();
        if (!source) return;

        try {
            loadBtn.textContent = '...';
            const response = await fetch('/set_source', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source })
            });
            const data = await response.json();
            
            if (data.status === 'success') {
                loadBtn.textContent = 'DONE';
                loadBtn.style.background = '#4ade80';
                // Refresh video feed by re-setting the src
                const videoFeed = document.querySelector('.video-feed');
                videoFeed.src = videoFeed.src.split('?')[0] + '?t=' + new Date().getTime();
                
                setTimeout(() => {
                    loadBtn.textContent = 'LOAD';
                    loadBtn.style.background = 'var(--primary)';
                }, 2000);
            }
        } catch (error) {
            console.error('Error setting source:', error);
            loadBtn.textContent = 'FAIL';
            loadBtn.style.background = '#ef4444';
        }
    });

    // Poll for stats every 5 seconds
    async function updateStats() {
        try {
            const response = await fetch('/stats');
            const data = await response.json();
            
            thresholdVal.textContent = data.threshold;
            modeVal.textContent = data.mode.toUpperCase() + ' AI';
        } catch (error) {
            console.error('Error fetching stats:', error);
            document.getElementById('conn-val').textContent = 'Error';
            document.getElementById('conn-val').style.color = '#ef4444';
        }
    }

    updateStats();  // Call on page load
    setInterval(updateStats, 5000);  // Poll every 5 seconds
    console.log('Crowd AI Pro Dashboard Initialized');
});
