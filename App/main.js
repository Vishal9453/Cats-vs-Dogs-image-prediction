$(document).ready(function() {
    // Get references to HTML elements
    var imageFileInput = document.getElementById('imageFile');
    var predictButton = document.getElementById('predictButton');
    var predictionResult = document.getElementById('predictionResult');

    // Add an event listener for the Predict button
    predictButton.addEventListener('click', async function() {
        // Get the selected image file
        var file = imageFileInput.files[0];
        if (!file) {
            // Display a message if no image is selected
            predictionResult.textContent = 'Please select an image.';
            return;
        }

        // Create a FormData object to send the image file to the server
        var formData = new FormData();
        formData.append('imagefile', file);

        try {
            // Send a POST request to the server for prediction
            var response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                // Display the prediction result if the request is successful
                var prediction = await response.text();
                predictionResult.textContent = 'Prediction: ' + prediction;
            } else {
                // Display an error message if the request is not successful
                var errorMessage = await response.text();
                predictionResult.textContent = 'Error: ' + errorMessage;
            }
        } catch (error) {
            // Handle errors and display an error message
            predictionResult.textContent = 'An error occurred.';
            console.error('Error:', error);
        }
    });
});
