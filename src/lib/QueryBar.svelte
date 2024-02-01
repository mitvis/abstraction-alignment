<script lang='ts'>
    import {computeEntropy} from './hierarchyUtils';
    import type { HierarchyNode } from "d3-hierarchy";
    import type { Node } from "./types";

    export let selectedIDs: number[];
    export let trees: HierarchyNode<Node>[];

    let dimensions = 0;
    $: if (trees && trees.length > 0) {
        const maxHeight = trees.reduce((acc, tree) => tree.height > acc ? acc = tree.height : acc = acc, 0);
        dimensions = maxHeight + 1;
    }
    let queryVector: string[] = [];
    $: queryVector = Array.from({length: dimensions}, () => '*');

    let thresholds: number[] = [];
    $: thresholds = Array.from({length: dimensions}, () => 0);

    function euclideanDistance(vector1: number[], vector2: number[]): number {
        let sum = 0;
        for (let i = 0; i < vector1.length; i++) {
            sum += Math.pow(vector1[i] - vector2[i], 2);
        }
        return Math.sqrt(sum);
    };

    function parseQuery(query: string, threshold: number) {
        let parser = (level: number[]): boolean => true;
        if (query.startsWith('!')) {
            let childParser = parseQuery(query.slice(1), threshold);
            parser = (level: number[]): boolean => {
                return ! childParser(level);
            }
        } else if (query.startsWith('[') && query.endsWith(']')) {
            query = query.slice(1, -1); // remove the brackets
            let queryValues = query.split(',').map(value => parseFloat(value));
            parser = (level: number[]): boolean => {
                let sortedLevelValues = level.sort((a, b) => a - b);
                let sortedQueryValues = queryValues.sort((a, b) => a - b);
                if (sortedQueryValues.length < sortedLevelValues.length) {
                    sortedQueryValues = [
                        ...sortedQueryValues, 
                        ...new Array(sortedLevelValues.length - sortedQueryValues.length).fill(0)
                    ];
                }
                return euclideanDistance(sortedLevelValues, sortedQueryValues) <= threshold;
            };
        } else if (!isNaN(parseInt(query))) {
            parser = (level: number[]): boolean => {
                return level.length === parseInt(query);
            };
        } else if (query.startsWith('>')) {
            let subQuery = query.slice(1);
            parser = (level: number[]): boolean => {
                return level.length > parseInt(subQuery);
            };
        } else if (query.startsWith('<')) {
            let subQuery = query.slice(1);
            parser = (level: number[]): boolean => {
                return level.length < parseInt(subQuery);
            };
        };
        return parser
    };

    function query(queryVector: string[], thresholds: number[]) {
        selectedIDs = trees
            .map((root, index) => {return {root: root, index: index}})
            .filter(object => {
                let levelNodes: number[][] = [];
                for (let level = 0; level <= object.root.height; level++) {
                    let levelValues: number[] = object.root.descendants()
                        .filter(node => node.depth === level)
                        .map(node => node.data.value);
                    if ( level > 0 && levelNodes[level-1].length  > levelValues.length ) {
                        const extraZeros = new Array(levelNodes[level-1].length - levelValues.length).fill(0);
                        levelValues = [...levelValues, ...extraZeros];
                    };
                    levelNodes.push(levelValues);
                };
                for (let level = 0; level <= object.root.height; level++) {
                    let parser = parseQuery(queryVector[level], thresholds[level]);
                    if (!parser(levelNodes[level])) {
                        return false;
                    }
                }
                return true;
            })
            .map(object => object.index);
    }

    $: if (queryVector.length > 0 && thresholds.length > 0) {
        query(queryVector, thresholds);
    }
</script>

<div id='query-vector'>
    <!-- create an input value for each {#each queryVector} -->
    <p>Query:</p>
    <div class='vector'>
        {#each queryVector as value, i}
            <div class='query'>
                {#if i === 0}
                    <span>Level {i} (Root)</span>
                {:else if i === queryVector.length - 1}
                    <span>Level {i} (Leaves)</span>
                {:else}
                    <span>Level {i}</span>
                {/if}
                <input class='query-dimension' value={queryVector[i]} on:keydown={(e) => {if (e.key === 'Enter') queryVector[i] = e.target.value}} />
            </div>
        {/each}
    </div>
    <button 
        id='query-button' 
        on:click={() => {
            queryVector = queryVector.map(_ => '*'); 
            thresholds = thresholds.map(_ => 0);
        }}
    >
        Clear
    </button>
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
