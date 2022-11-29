import React from 'react';
import Main from './Main';

import './App.css';
import { usePromiseTracker } from "react-promise-tracker";
import {TailSpin} from 'react-loader-spinner';


const LoadingIndicator = props => {
  const { promiseInProgress } = usePromiseTracker();

  return (
    promiseInProgress &&
    <div
      style={{
        width: "100%",
        height: "100",
        display: "flex",
        justifyContent: "center",
        alignItems: "center"
      }}
    >
      <TailSpin color="#2BAD60" height="100" width="100" />
    </div>
  );
}

const App = () => (
  <div>
    <h1>Anomaly Detection</h1>
    <Main />
    <LoadingIndicator/>
  </div>
)

export default App;
