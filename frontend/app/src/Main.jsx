import React from "react";
import { trackPromise } from 'react-promise-tracker';

class Main extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            imageFILE: null,
            imageLabel: '',
        }

        this.handleChange = this.handleChange.bind(this);
        this.handleCheckImage = this.handleCheckImage.bind(this);
    }

    handleCheckImage = event => {
        event.preventDefault();
        
        const data = new FormData();
        data.append('imageFile', this.uploadImage.files[0]);

        trackPromise(
            fetch('http://localhost:5000/detection', {
                method: 'POST',
                body: data,
            }).then((response) => {
                response.json().then((result) => {
                    this.setState({
                        imageLabel: result.label
                    })
                });
        }));
    }

    handleChange = event => {
        event.preventDefault();
        this.setState({
            imageFILE: URL.createObjectURL(event.target.files[0]) 
        })
    };

    render() {
        return (
            <div className="App">
                <header className="App-header">
                    <form onSubmit={this.handleCheckImage}>
                        <label>
                            <input ref={(ref) => {this.uploadImage = ref; }} type="file" onChange={this.handleChange} />
                            Select File
                        </label>
                        <br/>
                        <br/>
                        <div>
                            <img src={this.state.imageFILE} width="255" height="255" />
                        </div>
                        <br/>
                        <label>
                            <input type="submit" value="Check" />
                            Check Anomaly
                        </label>
                        <p>{this.state.imageLabel}</p>
                    </form>
                </header>
            </div>
        );
    }
}

export default Main;