
import React from "react";
import ProbChart from "./pages/reactChart";
import { backdropClasses } from "@mui/material";

const ResultDiv = ({ result, dataPoints }) => {
  return (
    <div style={styles.resultDiv}>
      <div>
        <div style={styles.result}>
          <h1>Most Likely Outcome</h1>
          <div style={styles.resultContent}>
            <div>
              <h3>RESULT: {result && result.class}</h3>
            </div>
            <div>
              <p>Probability: {dataPoints && dataPoints.map(a =>{
                if(a.label === result.class){
                  return a.y.toFixed(2) * 100 + "%"
                }
              })}</p>
            </div>
          </div>
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        ></div>
      </div>
      <div style={styles.chart}>
        {dataPoints && <ProbChart dataPoints={dataPoints} />}
      </div>
    </div>
  );
};

export default ResultDiv;

const styles = {
    result: {
        display: "flex",
        width: "100%",
        flexDirection: "column",
        // justifyContent: "center",
        alignItems: "center",
        backgroundColor: "#f2f2f2",
        borderRadius: "10px",
        marginBottom: 20,
        padding: 10,
        boxShadow: "0px 2px 3.84px rgba(0,0,0,0.5)",
      },
      resultContent: {
        display: "flex",
        width: "70%",
        justifyContent: "space-between",
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: "#f2f2f2",
      },
      chart: {
        width: "104%",
        boxShadow: "0px 10px 10px rgba(0, 0, 0, 0.5)",
        elevation: 2,
      },
      resultDiv: {
        display: "flex",
        justifyContent: "center",
        // alignItems: "center",
        flexDirection: "column",
        width: "40%",
        height: "90%",
        paddingLeft: "5%",
        paddingRight: "5%",
        borderRadius: "10px",
        margin: 20,
        background: "linear-gradient(145deg, rgba(0, 0, 128,0.6), #000080) ",
        // backgroundColor:'#000066',
        boxShadow: "0px 2px 3.84px rgba(0,0,0,0.5)",
      },
}