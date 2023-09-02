$(document).ready(function() {
    var imageFileInput = document.getElementById('imageFile');
    var predictButton = document.getElementById('predictButton');
    var predictionResult = document.getElementById('predictionResult');

    predictButton.addEventListener('click', async function() {
        var file = imageFileInput.files[0];
        if (!file) {
            predictionResult.textContent = 'Please select an image.';
            return;
        }

        var formData = new FormData();
        formData.append('imagefile', file);

        try {
            var response = await fetch('http://0.0.0.0:10000/predict', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                var prediction = await response.text();
                predictionResult.textContent = 'Prediction: ' + prediction;
            } else {
                var errorMessage = await response.text();
                predictionResult.textContent = 'Error: ' + errorMessage;
            }
        } catch (error) {
            predictionResult.textContent = 'An error occurred.';
            console.error('Error:', error);
        }
    });
});
