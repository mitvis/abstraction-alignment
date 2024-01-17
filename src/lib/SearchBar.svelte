<script lang='ts'>
    import {computeEntropy} from './hierarchyUtils';

    export let selectedIDs: number[];
    let selectedID: number;
    
</script>

<div id='query-vector'>
    <p>Search ID:</p>
    <div class='vector'>
        <input type='number' class='query-dimension' bind:value={selectedID} />
    </div>
    <button id='query-button' on:click={() => selectedIDs=[...selectedIDs, selectedID]}>Search</button>
    <button id='query-button' on:click={() => selectedIDs=[]}>Clear</button>
</div>

<style>
    #query-vector {
        display: flex;
        flex-direction: row;
        column-gap: 1em;
        align-items: flex-end;
    }

    .query-dimension {
        border: 0;
        outline: 0;
        box-shadow: rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(60 66 87 / 16%) 0px 0px 0px 1px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px;
        border-radius: 4px;
        padding: 4px 8px;
        width: 100%;
        box-sizing: border-box;
    }
    .query-dimension:focus{
        box-shadow: rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(58 151 212 / 36%) 0px 0px 0px 2px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(60 66 87 / 16%) 0px 0px 0px 1px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px;
    }

    .query {
        display: flex;
        flex-direction: column;
        align-items: start;
        width: 10rem;
        row-gap: 0.2em;
    }
    
    .query span {
        color: grey;
        font-variant-caps: all-small-caps;
    }

    .vector {
        display: flex;
        flex-direction: row;
        column-gap: 0.5em;
    }

    #query-button {
        margin-top: 1.4em;
        height: 2.1em;
        border-radius: 4px;
    }

    p {
        font-size: 12pt;
        font-weight: 600;
        margin-bottom: 0;
    }

    input[type=range] {
        width: 100%;
        height: 5px; /* Specified height */
        -webkit-appearance: none; /* Override default CSS styles */
        appearance: none;
        background: #d3d3d3; /* Grey background */
        outline: none; /* Remove outline */
        border-radius: 2.5px;
        margin: 0;
        margin-top: 0.2em;
        margin-bottom: -0.1em;
    }

    /* Thumb styles for Chrome, Safari, Edge, Opera */
    input[type=range]::-webkit-slider-thumb {
        -webkit-appearance: none; /* Override default look */
        appearance: none;
        width: 8px; /* Set a specific slider handle width */
        height: 8px; /* Slider handle height */
        background: grey; /* Grey background */
        border-radius: 50%; /* Make it circular */
        margin-top: -1px;
    }

    /* Thumb styles for Firefox */
    input[type=range]::-moz-range-thumb {
        width: 8px; /* Set a specific slider handle width */
        height: 8px; /* Slider handle height */
        background: grey; /* Grey background */
        border-radius: 50%; /* Make it circular */
        margin-top: -1px;
    }

    /* Track styles for Firefox */
    input[type=range]::-moz-range-track {
        width: 100%;
        height: 5px;
        background: #d3d3d3; /* Grey background */
        border-radius: 2.5px;
    }

    /* Track styles for Chrome, Safari, Edge, Opera */
    input[type=range]::-webkit-slider-runnable-track {
        width: 100%;
        height: 5px;
        background: #d3d3d3; /* Grey background */
        border-radius: 2.5px;
    }
    
</style>
