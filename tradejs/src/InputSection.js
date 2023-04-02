import React, {useEffect, useState} from "react";
import RangeInput from "./RangeInput";
import CargoInput from "./CargoInput";
import CommodityInput from "./CommodityInput";
import LocationInput from "./LocationInput";
import StepInput from "./StepInput";
import {Fade, Form} from "react-bootstrap";
import Notification from "./Notification";


const commodity_url = "/commodities"
const location_url = "/locations"


function InputSection({onSubmit, onBlacklistReset, onBlacklist, lockForm}) {
    const [range, setRange] = useState(2);
    const [step, setStep] = useState(2);
    const [cargo, setCargo] = useState(696);
    const [commodities, setCommodities] = useState([]);
    const [locations, setLocations] = useState([]);
    const [comOptions, setComOptions] = useState([])
    const [locOptions, setLocOptions] = useState([])
    const [showBlkNotification, setShowBlkNotification] = useState(false);
    const [showResetNotification, setResetNotification] = useState(false);

    useEffect(() => {
        // fetch data
        const dataFetch = async (url) => {
            return await (
                await fetch(
                    url
                )
            ).json()
        };

        dataFetch(commodity_url).then(d => setComOptions(d))
        dataFetch(location_url).then(d => setLocOptions(d))
    }, []);

    const handleRangeChange = (value) => {
        setRange(value);
    };

    const handleStepChange = (value) => {
        setStep(value);
    };

    const handleCargoChange = (value) => {
        setCargo(value);
    };

    const handleCommodityChange = (index, name, value) => {
        const newCommodities = [...commodities];
        newCommodities[index].amount = value;
        newCommodities[index].name = name;
        setCommodities(newCommodities);
    };

    const handleCommodityRemove = (index) => {
        const newCommodities = [...commodities];
        newCommodities.splice(index, 1);
        setCommodities(newCommodities);
    };

    const handleLocationChange = (index, name) => {
        const newLocations = [...locations];
        newLocations[index] = name;
        setLocations(newLocations);
    };

    const handleLocationRemove = (index) => {
        const newLocations = [...locations];
        newLocations.splice(index, 1);
        setLocations(newLocations);
    };

    const handleAddLocation = () => {
        const newLocations = [...locations, ""];
        setLocations(newLocations);
    }

    const handleAddCommodity = () => {
        const newCommodities = [...commodities, {name: "", amount: 100}];
        setCommodities(newCommodities);
    };

    const validateForm = (form) => {
        if (!form.checkValidity()) {
            return false
        }
        for (const c of commodities) {
            if (!comOptions.includes(c.name)) {
                return false
            }
        }
        for (const l of locations) {
            if (!locOptions.includes(l)) {
                return false
            }
        }
        return true
    }

    const handleSubmit = (event) => {
        event.preventDefault();
        const form = event.currentTarget;
        if (validateForm(form) === false) {
            event.stopPropagation();
            return
        }
        setShowBlkNotification(false)
        setResetNotification(false)
        onSubmit(range, step, cargo, commodities, locations);
    };

    const handleBlacklist = () => {
        onBlacklist()
        setShowBlkNotification(true)
    };

    const handleBlacklistReset = () => {
        onBlacklistReset();
        setResetNotification(true)
    };

    return (
        <Form onSubmit={handleSubmit} noValidate>
            <div className="form-row mb-3">
                <div className="col-md-6">
                    <RangeInput value={range} onChange={handleRangeChange}/>
                </div>
                <div className="col-md-6">
                    <StepInput value={step} onChange={handleStepChange}/>
                </div>
                <div className="col-md-6">
                    <CargoInput value={Number(cargo)} onChange={handleCargoChange}/>
                </div>
            </div>

            {commodities.map((commodity, index) => (
                <CommodityInput
                    key={index}
                    index={index}
                    value={commodity.amount}
                    name={commodity.name}
                    options={comOptions}
                    onChange={handleCommodityChange}
                    onRemove={handleCommodityRemove}
                />
            ))}

            {locations.map((location, index) => (
                <LocationInput
                    key={index}
                    index={index}
                    value={location}
                    options={locOptions}
                    onChange={handleLocationChange}
                    onRemove={handleLocationRemove}
                />
            ))}

            <div className="form-row mb-3">
                <div className="col-md-6">
                    <div className="btn-group">
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={handleAddCommodity}
                        >
                            Max Commodity
                        </button>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={handleAddLocation}
                        >
                            Remove Location
                        </button>
                        <button
                            type="button"
                            className="btn btn-warning"
                            onClick={handleBlacklist}
                            disabled={lockForm}
                        >
                            Blacklist Plan
                        </button>
                        <button
                            type="button"
                            className="btn btn-danger"
                            onClick={handleBlacklistReset}
                            disabled={lockForm}
                        >
                            Reset Blacklist
                        </button>
                    </div>
                </div>
            </div>

            <div className="form-row mb-4">
                <div className="col-md-12">
                    <button type="submit" className="btn btn-primary btn-block" disabled={lockForm}>
                        Submit
                    </button>
                </div>
            </div>

            <div className="form-row col-md-7 mb-4">
                {showBlkNotification && (
                    <Notification
                        message={"Plan Blacklisted"}
                        onClose={() => setShowBlkNotification(false)}
                        type={"success"}
                    />
                )}
                {showResetNotification && (
                    <Notification
                        message={"Blacklist cleared"}
                        onClose={() => setResetNotification(false)}
                        type={"success"}
                    />
                )}
            </div>
        </Form>

    );

}


export default InputSection;
