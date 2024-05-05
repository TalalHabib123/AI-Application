import robo from "./2.png";
import { FaExclamationTriangle } from "react-icons/fa";
import { useState } from "react";
import {Bar } from 'react-chartjs-2';
import ProbChart from "./reactChart";


const Result = () => {
  const [result, setResult] = useState(null);


  return (
    <div style={{ height: "100%", margin: 5 }}>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div
          style={{
            backgroundColor: "#176B87",
            padding: 10,
            width: "80%",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            borderRadius: "10px",
          }}
        >
            <h3>Your Results</h3>
            <h4>Class: </h4><p>Melanoma</p>
            <ProbChart/>
        </div>
      </div>
    </div>
  );
};

const NoResult = () => {
  return (
    <>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div
          style={{
            backgroundColor: "#176B87",
            padding: 10,
            width: "80%",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            borderRadius: "10px",
          }}
        >
          {/* <h2>Your Results</h2> */}
          <h1>LesionLabs AI</h1>
          <img src={robo} style={{ width: "75%" }}></img>
        </div>

        <div
          style={{
            backgroundColor: "#DDDDDD",
            padding: 20,
            width: "60%",
            display: "flex",
            flexDirection: "column",
            margin: 10,
            borderRadius: 5,
            border: "1px solid black",
            color: "black",
          }}
        >
          <h4 style={{ margin: 2 }}>DISCLAIMER:</h4>
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            <div style={{ marginRight: 10 }}>
              <FaExclamationTriangle size={40} />
            </div>
            <div>
              <p>
                {" "}
                This is a research project and not a substitute for professional
                medical advice. Please consult a doctor for any medical
                concerns.
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Result;
