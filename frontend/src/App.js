import "./App.css";
import axios from "axios";
import { useState } from "react";
import { FaCheck, FaUpload } from "react-icons/fa";
import { FaCamera } from "react-icons/fa";
import { FaArrowLeft } from "react-icons/fa";
import { FaEdit } from "react-icons/fa";
import { FaExclamationTriangle } from "react-icons/fa";

import Webcam from "react-webcam";
import ResultDiv from "./ResultDiv";

import roboto from "./Roboto.png";

function dataURLtoFile(dataurl, filename) {
  let arr = dataurl.split(","),
    mime = arr[0].match(/:(.*?);/)[1],
    bstr = atob(arr[1]),
    n = bstr.length,
    u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, { type: mime });
}

const labels = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"];
const App = () => {
  const [imageSrc, setImgSrc] = useState("");
  const [imgSourceButton, setImgSrcBtn] = useState("");
  const [img, setImg] = useState("");
  const [chatHover, setChatHover] = useState("");
  const [file, setFile] = useState(null);
  const [dataPoints, setDataPoints] = useState(null);
  const [result, setResult] = useState(null);

  const evaluate = async () => {
    const url = "http://localhost:5000/predict";
    let newfile;
    const formData = new FormData();
    if (file) {
      console.log(file);
      formData.append("file", file);
    } else {
      newfile = dataURLtoFile(img, "image.jpg");
      formData.append("file", newfile);
    }

    const response = await axios.post(url, formData, {
      headers: {
        "Content-Type": `multipart/form-data; boundary=${formData._boundary}`,
      },
    });
    let res = { class: response.data.class, label: response.data.label };
    let dataPoints = response.data.predictions[0].map((element, index) => {
      return { y: element, label: labels[index] };
    });
    setDataPoints(dataPoints);
    setResult(res);
  };
  const hoverTrue = (e) => {
    setImgSrcBtn(e.target.value);
  };
  const hoverFalse = (e) => {
    setImgSrcBtn(null);
  };
  const onFileChange = (e) => {
    setImgSrc("uploaded");
    setFile(e.target.files[0]);
    setImg(URL.createObjectURL(e.target.files[0]));
  };
  const setUploadSrc = (e) => {
    setImgSrc(e.target.value);
  };

  return (
    <>
      <div style={styles.container}>
        {result && dataPoints && (
          <>
            <ResultDiv result={result} dataPoints={dataPoints} />
            <div style={styles.addImage}>
              <div style={styles.noImageSelect}>
                <img src={img} width="90%" height={400} />
                <div style={styles.uploadImgButtons}>
                  <button
                    value=""
                    style={imgSourceButton == "" ? styles.btnHover : styles.btn}
                    onMouseEnter={hoverTrue}
                    onMouseLeave={hoverFalse}
                    onClick={() => {
                      setImgSrc("");
                      setDataPoints(null);
                      setResult(null);
                      setImg("");
                      setFile(null);
                    }}
                  >
                    <div style={{ marginBottom: "5px" }}>
                      <FaEdit size={50} />
                    </div>
                    Change Photo
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
        {!result && !dataPoints && (
          <div style={styles.addImage}>
            { imageSrc == "UseCam" ? (
              <div style={styles.noImageSelect}>
                <div
                  style={{
                    boxShadow: "0px 2px 3.84px rgba(0,0,0,0.25)",
                    borderRadius: "10px",
                    padding: "3%",
                    margin: "5%",
                  }}
                >
                  <Webcam
                    audio={false}
                    screenshotFormat="image/jpeg"
                    width={500}
                    mirrored
                  >
                    {({ getScreenshot }) => (
                      <div style={styles.uploadImgButtons}>
                        <button
                          style={
                            imgSourceButton == "cancel"
                              ? styles.btnHover
                              : styles.btn
                          }
                          paddingRight={10}
                          paddingLeft={10}
                          value="cancel"
                          onMouseEnter={hoverTrue}
                          onMouseLeave={hoverFalse}
                          onClick={() => {
                            setImgSrc("");
                          }}
                        >
                          <FaArrowLeft size={40} />
                          Cancel
                        </button>
                        <button
                          value="img"
                          style={
                            imgSourceButton == "img"
                              ? styles.btnHover
                              : styles.btn
                          }
                          onMouseEnter={hoverTrue}
                          onMouseLeave={hoverFalse}
                          paddingRight={10}
                          paddingLeft={10}
                          onClick={() => {
                            setImgSrc("img");
                            setImg(getScreenshot());
                            setImgSrc("uploaded");
                          }}
                        >
                          <FaCamera size={40} />
                          Capture
                        </button>
                      </div>
                    )}
                  </Webcam>
                </div>
              </div>
            ) : imageSrc == "UploadImg" ? (
              <div style={styles.noImageSelect}>
                <h3 style={styles.noImageSelectText}>
                  Select an image to evaluate
                </h3>
                <div style={styles.uploadImgButtons}>
                  <button
                    value="UploadImg"
                    style={
                      imgSourceButton == "UploadImg"
                        ? styles.btnHover
                        : styles.btn
                    }
                    onMouseEnter={hoverTrue}
                    onMouseLeave={hoverFalse}
                    onClick={setUploadSrc}
                  >
                    <div style={{ marginBottom: "5px" }}>
                      <FaUpload size={50} />
                    </div>
                    Upload Image
                  </button>
                  <button
                    value="UseCam"
                    style={
                      imgSourceButton == "UseCam" ? styles.btnHover : styles.btn
                    }
                    onMouseEnter={hoverTrue}
                    onMouseLeave={hoverFalse}
                    onClick={setUploadSrc}
                  >
                    <div style={{ marginBottom: "5px" }}>
                      <FaCamera size={50} />
                    </div>
                    Use Camera
                  </button>
                </div>
                <input type="file" onChange={onFileChange}></input>
              </div>
            ) : imageSrc == "uploaded" ? (
              <div style={styles.noImageSelect}>
                <img src={img} width="90%" height={400} />
                <div style={styles.uploadImgButtons}>
                  <button
                    value=""
                    style={imgSourceButton == "" ? styles.btnHover : styles.btn}
                    onMouseEnter={hoverTrue}
                    onMouseLeave={hoverFalse}
                    onClick={setUploadSrc}
                  >
                    <div style={{ marginBottom: "5px" }}>
                      <FaEdit size={50} />
                    </div>
                    Change Photo
                  </button>
                  <button
                    value="UseCam"
                    style={
                      imgSourceButton == "UseCam" ? styles.btnHover : styles.btn
                    }
                    onMouseEnter={hoverTrue}
                    onMouseLeave={hoverFalse}
                    onClick={evaluate}
                  >
                    <div style={{ marginBottom: "5px" }}>
                      <FaCheck size={50} />
                    </div>
                    Evaluate
                  </button>
                </div>
              </div>
            ) : (
              <div style={styles.noImageSelect}>
                <img src={roboto} width="90%" />
                <h2 style={styles.noImageSelectText}>
                  Select an image to evaluate
                </h2>
                <div style={styles.uploadImgButtons}>
                  <button
                    value="UploadImg"
                    style={
                      imgSourceButton == "UploadImg"
                        ? styles.btnHover
                        : styles.btn
                    }
                    onMouseEnter={hoverTrue}
                    onMouseLeave={hoverFalse}
                    onClick={setUploadSrc}
                  >
                    <div style={{ marginBottom: "5px" }}>
                      <FaUpload size={50} />
                    </div>
                    Upload Image
                  </button>
                  <button
                    value="UseCam"
                    style={
                      imgSourceButton == "UseCam" ? styles.btnHover : styles.btn
                    }
                    onMouseEnter={hoverTrue}
                    onMouseLeave={hoverFalse}
                    onClick={setUploadSrc}
                  >
                    <div style={{ marginBottom: "5px" }}>
                      <FaCamera size={50} />
                    </div>
                    Use Camera
                  </button>
                </div>
                <div style ={{display:'flex', flexDirection:'column', justifyContent:'center', alignItems:'center'
                }}>
                  <div style ={{display:'flex', justifyContent:'center', alignItems:'center', marginBottom:'0px'}}> 
                    <FaExclamationTriangle size={30} />
                    <p >! DISCLAIMER ! </p>
                  </div>
                  <div>
                  <p>
                    The information provided here is intended for general
                    knowledge and educational purposes only. It is not a
                    substitute for professional medical advice, diagnosis, or
                    treatment.
                  </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
};

export default App;

const styles = {
  container: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    height: "100vh",
    background:
      "url(https://images.onlymyhealth.com/imported/images/2023/September/25_Sep_2023/main_CancerYoungAdults.jpg) no-repeat center center/cover",
  },
  noImageSelectText: {
    fontSize: 20,
    fontWeight: "bold",
    color: "black",
    marginBottom: "10px",
  },
  addImage: {
    display: "flex",
    justifyContent: "center",
    // alignItems: "center",
    flexDirection: "column",
    width: "40%",
    height: "90%",
    paddingLeft: "5%",
    paddingRight: "5%",
    backgroundColor: "rgba(169, 169, 169, 0.6)",
    borderRadius: "10px",
    margin: 20,
    boxShadow: "0px 2px 3.84px rgba(0,0,0,0.5)",
  },
  noImageSelect: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    width: "100%",
  },
  uploadImgButtons: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "row",
    width: "100%",
  },
  btn: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    background: "linear-gradient(rgb(51, 51, 204,1))",
    backgroundColor: "#131386	",
    border: "1px solid white",
    borderRadius: "5px",
    padding: "3%",
    margin: "5%",
    color: "white",
    boxShadow: "0px 2px 3.84px rgba(0,0,0,0.25)",
    fontSize: 15,
    fontWeight: "bold",
    // transition: "0.01s ",
  },
  camBtn: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#C4BFBF",
    background: "None",
    border: "1px solid white",
    borderRadius: "5px",
    padding: "3%",
    bordercolor: "black",
    margin: "2% 2% 2% 15%",
    color: "white",
  },
  btnHover: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    background: "linear-gradient(rgb(71, 71, 209,1),rgb(51, 51, 204,1))",
    backgroundColor: "#131386	",
    border: "1px solid white",
    borderRadius: "5px",
    padding: "3%",
    margin: "5%",
    color: "white",
    boxShadow: "0px 2px 3.84px rgba(0,0,0,0.25)",
    fontSize: 15,
    fontWeight: "bold",
    transition: "ease-in-out",
  },
};
//     height: "100vh",
//     // background:'url(https://images.unsplash.com/photo-1631556097152-c39479bbff93?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D) no-repeat center center/cover'

// },
// linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
