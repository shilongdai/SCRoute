import React, {useEffect, useState} from 'react';
import {Typeahead} from 'react-bootstrap-typeahead';
import 'react-bootstrap-typeahead/css/Typeahead.css';


const location_url = "/locations"


function LocationInput({key, index, value, options, onChange, onRemove}) {

    const handleLocationChange = (val) => {
        onChange(index, val[0]);
    };

    return (
        <div key={key} className="row mb-3 d-flex justify-content-between">
            <Typeahead
                className="location-select col-md-8"
                placeholder="Location"
                selected={options.includes(value) ? [value] : []}
                onChange={handleLocationChange}
                options={options}
                {...options.includes(value) ? {} : {isInvalid: true}}
                disabled={options.length === 0}
                required={true}
            />
            <div className="col-md-4 d-flex mr-2 align-items-end">
                <button type="button" className="btn btn-danger" onClick={() => onRemove(index)}>
                    Remove
                </button>
            </div>
        </div>
    );
}

export default LocationInput;
