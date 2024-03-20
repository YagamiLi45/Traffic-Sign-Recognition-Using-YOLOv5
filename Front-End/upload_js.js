const form = document.querySelector("form"),
    fileInput = document.querySelector(".file-input"),
    progressArea = document.querySelector(".progress-area"),
    uploadedArea = document.querySelector(".uploaded-area");

form.addEventListener("click", () => {
    fileInput.click();
});

fileInput.onchange = ({target}) => {
    let file = target.files[0];
    if (file) {
        uploadFile(file);
    }
}

function uploadFile(file) {
    let xhr = new XMLHttpRequest();
    xhr.open("POST", "http://127.0.0.1:5000/predict");
    xhr.upload.addEventListener("progress", ({loaded, total}) => {
        let fileLoaded = Math.floor((loaded / total) * 100);
        let progressHTML = `<li class="row">
                              <i class="fas fa-file-alt"></i>
                              <div class="content">
                                <div class="details">
                                  <span class="name">${file.name} • Uploading</span>
                                  <span class="percent">${fileLoaded}%</span>
                                </div>
                                <div class="progress-bar">
                                  <div class="progress" style="width: ${fileLoaded}%"></div>
                                </div>
                              </div>
                            </li>`;
        uploadedArea.classList.add("onprogress");
        progressArea.innerHTML = progressHTML;
        if (loaded === total) {
            progressArea.innerHTML = "";
            let uploadedHTML = `<li class="row">
                                  <div class="content upload">
                                    <i class="fas fa-file-alt"></i>
                                    <div class="details">
                                      <span class="name">${file.name} • Uploaded</span>
                                    </div>
                                  </div>
                                  <i class="fas fa-check"></i>
                                </li>`;
            uploadedArea.classList.remove("onprogress");
            uploadedArea.insertAdjacentHTML("afterbegin", uploadedHTML);

            // Display the result image
            let response = JSON.parse(xhr.responseText);
            displayResultImage(response.result_image);
        }
    });

    let data = new FormData();
    data.append('file', file);
    xhr.send(data);
}

function displayResultImage(resultImageBase64) {
    let resultHTML = `<div class="result">
                        <h2>Result Image</h2>
                        <img src="data:image/jpeg;base64,${resultImageBase64}" alt="Result Image">
                      </div>`;
    uploadedArea.insertAdjacentHTML("beforeend", resultHTML);
}
