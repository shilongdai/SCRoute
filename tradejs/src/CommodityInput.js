import React, {useEffect, useState} from 'react';
import {Typeahead} from 'react-bootstrap-typeahead';
import 'react-bootstrap-typeahead/css/Typeahead.css';


const commodity_url = "/commodities"


function CommodityInput({index, value, name, options, onChange, onRemove}) {

    const handleNameChange = (newName) => {
        newName = newName[0]
        onChange(index, newName, value);
    };
    const handleValueChange = (event) => {
        const newVal = Number(event.target.value);
        onChange(index, name, newVal);
    };
    return (
        <div key={index} className="commodity-input row mb-3">
            <Typeahead name="name" placeholder={"Commodity"}
                       selected={options.includes(name) ? [name] : []} onChange={handleNameChange}
                       className="commodity-select col-md-4"
                       {...options.includes(name) ? {valid: true} : {isInvalid: true}}
                       disabled={options.length === 0}
                       required={true} options={options}
            />
            <div className="col-md-4">
                <div className="input-group">
                    <input type="number" name="amount" value={value} max={100} min={0}
                           onChange={handleValueChange} className="form-control"/>
                    <div className="input-group-append">
                        <span className="input-group-text">%</span>
                    </div>
                </div>
            </div>
            <div className="col-md-4 d-flex align-items-center">
                <button type="button" onClick={() => onRemove(index)} className="btn btn-danger">
                    Remove
                </button>
            </div>
        </div>
    );

}

export default CommodityInput;
