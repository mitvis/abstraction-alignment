<script lang='ts'>
    export let selectedIDs: number[];
    export let embeddings: number[][];

    $: dimensions = embeddings.length > 0 ? embeddings[0].length : 0;

    let queryVector: string[];
    $: queryVector = Array.from({length: dimensions}, () => '*');

    let thresholds: number[];
    $: thresholds = Array.from({length: dimensions}, () => 0);

    function computeEntropy(probabilities: number[]) {
        let entropy = probabilities.reduce((entropy, probability) => {
            if (probability > 0) {
                return entropy - probability * Math.log(probability);
            } else {
                return entropy;
            }
        }, 0);
        return entropy;
    }

    function parseQuery(query: string, threshold: number) {
        if (query === '*') { 
            let parser = (value: number): boolean => true;
            return parser;
        }
        if (!isNaN(parseFloat(query))) {
            let queryNumber = parseFloat(query);
            let parser = (value: number): boolean => value >= queryNumber - threshold && value <= queryNumber + threshold;
            return parser;
        }
        if (query.startsWith('not')) {
            let subQuery = query.slice(4, -1) // remove the not and brackets
            let parser = (value: number): boolean => ! parseQuery(subQuery, threshold)(value);
            return parser;
        }
        if (query.startsWith('E')) {
            let subQuery = query.slice(2, -1) // remove the entropy and brackets
            let probabilities = subQuery.split(',').map(value => parseFloat(value));
            let entropy = computeEntropy(probabilities);
            return parseQuery(entropy.toString(), threshold);

        }
        if (query.startsWith('>')) {
            let subQuery = query.slice(1);
            let queryNumber = parseFloat(subQuery);
            let parser = (value: number): boolean => value > queryNumber;
            return parser;
        }
        if (query.startsWith('<')) {
            let subQuery = query.slice(1);
            let queryNumber = parseFloat(subQuery);
            let parser = (value: number): boolean => value < queryNumber;
            return parser;
        }
        throw new Error(`Invalid query: ${query}`);
    }

    function query() {
        // filter embeddings so that selectedIDs only contains the IDs of the embeddings that match the query vector
        selectedIDs = embeddings
            .map((embedding, index) => {
                return {
                    embedding: embedding,
                    index: index
                }
            })
            .filter(object => {
                return object.embedding.every((value, i) => {
                    let parser = parseQuery(queryVector[i], thresholds[i]);
                    return parser(value);
                })
            })
            .map(embedding => embedding.index);
    }
</script>

<div id='query-vector'>
    <!-- create an input value for each {#each queryVector} -->
    <p>Query vector:</p>
    <div class='vector'>
        {#each queryVector as value, i}
            <div class='query'>
                <span>Level {i}</span>
                <input class='query-dimension' bind:value={queryVector[i]} />
                <input type='range' bind:value={thresholds[i]} min=0 max=1 step=0.05/>
                <span>{thresholds[i]}</span>
            </div>
        {/each}
    </div>
    <button id='query-button' on:click={query}>Query</button>
</div>

<style>
    #query-vector {
        display: flex;
        flex-direction: row;
        column-gap: 1em;
        align-items: start;
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
