import React, {useEffect, useState} from "react";
import RangeInput from "./RangeInput";
import CargoInput from "./CargoInput";
import CommodityInput from "./CommodityInput";
import LocationInput from "./LocationInput";
import StepInput from "./StepInput";
import {Form} from "react-bootstrap";
import CommodityLocationInput from "./CommodityLocationInput";


const commodity_url = "/commodities"
const location_url = "/locations"


function InputSection({onSubmit, onBlacklist, lockForm, highLevelPlan}) {
    const [range, setRange] = useState(2);
    const [step, setStep] = useState(2);
    const [cargo, setCargo] = useState(696);
    const [commodities, setCommodities] = useState([]);
    const [locations, setLocations] = useState([]);
    const [comOptions, setComOptions] = useState([])
    const [locOptions, setLocOptions] = useState([])
    const [restrictions, setRestrictions] = useState([])

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

    const handleComLocChange = (index, com, loc, val) => {
        const newRestrictions = [...restrictions]
        newRestrictions[index].commodity = com;
        newRestrictions[index].location = loc
        newRestrictions[index].value = val
        setRestrictions(newRestrictions)
    };

    const handleAddComLoc = () => {
        const newRestrictions = [...restrictions, {commodity: "", location: "", value: 100}]
        setRestrictions(newRestrictions)
    }

    const handleRemoveComLoc = (index) => {
        const newRestrictions = [...restrictions]
        newRestrictions.splice(index, 1);
        setRestrictions(newRestrictions)
    }

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
        for(const r of restrictions) {
            if (!locOptions.includes(r.location)) {
                return false
            }
            if (!comOptions.includes(r.commodity)) {
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
        onSubmit(range, step, cargo, commodities, locations, restrictions);
    };

    const addBlacklist = (newRestrictions, transactions) => {
        if (transactions) {
            for(const t of transactions) {
                newRestrictions.push({location: t.location, commodity: t.commodity, value: 0})
            }
        }
    }

    const handleBlacklist = () => {
        const newRestrictions = [...restrictions]
        if (highLevelPlan) {
            addBlacklist(newRestrictions, highLevelPlan.buyTransactions)
            addBlacklist(newRestrictions, highLevelPlan.sellTransactions)
        }
        setRestrictions(newRestrictions)
    }

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

            {restrictions.map((restriction, index) => (
                <CommodityLocationInput
                    key={index}
                    index={index}
                    value={restriction.value}
                    commodity={restriction.commodity}
                    location={restriction.location}
                    onChange={handleComLocChange}
                    onRemove={handleRemoveComLoc}
                    com_options={comOptions}
                    loc_options={locOptions}
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
                            Commodity
                        </button>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={handleAddLocation}
                        >
                            Location
                        </button>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={handleAddComLoc}
                        >
                            Tuning
                        </button>
                        <button
                            type="button"
                            className="btn btn-warning"
                            onClick={handleBlacklist}
                            disabled={lockForm}
                        >
                            Blacklist
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
        </Form>

    );

}


export default InputSection;
