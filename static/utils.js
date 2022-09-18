function loadImg(){
    // Show preview image for select file before upload
    $('#imagePreview').attr('src', URL.createObjectURL(event.target.files[0]));
}

// Upload image using ajax
$('#upload').click(function(){
    // Create form data
    var formData = new FormData();
    // add file to form data
    formData.append('file', $('#fileInput')[0].files[0]);
    $.ajax({
        url: '/api/upload', // API Endpoint
        type: 'POST', // Request type
        data: formData, // Request data
        contentType: false,
        processData: false,
        success: function(data){
            // On request succss, we show image from server
            $('#imagePreview').attr('src', data);
        }
    });
});
