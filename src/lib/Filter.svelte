<script lang='ts'>
    export let selectedIDs: number[];
    export let labels: string[];

    let selectedLabel: string;

    $: uniqueLabels = Array.from(new Set(labels)).sort();
    $: selectionOptions = ['all', ...uniqueLabels];
    $: selectedLabel = 'all';
    
    function filter() {
        // filter embeddings so that selectedIDs only contains the IDs of the embeddings that match the query vector
        selectedIDs = labels.map((label, i) => {
            return {'label': label, 'index': i}
        }).filter(item => selectedLabel == selectionOptions[0] || item.label == selectedLabel).map(item => item.index)
    }
</script>

<div id='filter'>
    <p>Filter:</p>
    <select class='filter-options' bind:value={selectedLabel} on:change={() => filter()}>
        {#each selectionOptions as label}
            <option value={label}>{label}</option>
        {/each}
    </select>
</div>

<style>
    #filter {
        display: flex;
        flex-direction: row;
        column-gap: 1em;
        align-items: end;
        margin-bottom: 1em;
        width: 100%;
        max-width: 650px;
    }

    .filter-options {
        border: 0;
        outline: 0;
        box-shadow: rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(60 66 87 / 16%) 0px 0px 0px 1px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px;
        border-radius: 4px;
        padding: 4px 8px;
        width: 80%;
        box-sizing: border-box;
    }
    .filter-options:focus{
        box-shadow: rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(58 151 212 / 36%) 0px 0px 0px 2px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(60 66 87 / 16%) 0px 0px 0px 1px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px, rgb(0 0 0 / 0%) 0px 0px 0px 0px;
    }

    p {
        font-size: 12pt;
        font-weight: 600;
        margin-bottom: 0;
    }
    
</style>
