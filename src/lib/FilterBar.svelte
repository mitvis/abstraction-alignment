<script lang='ts'>

    import type { Filter } from './types';

    type Option = {
        [key: string]: string[]
    }

    export let options: Option = {
        'nodeA': ['height', 'depth', 'name'],
        'nodeB': ['height', 'depth', 'name'],
        'pair': ['connected', 'shareParent']
    }
    export let filters: Filter[] = [];

    let selectedObject: string;
    let selectedAttribute: string;
    let selectedOperator: string;
    let operators = ['==', '!=', '<', '<=', '>', '>='];
    let selectedValue: string = 'Value'

    function addFilter() {
        const filter = {
            'object': selectedObject,
            'attribute': selectedAttribute,
            'operator': selectedOperator,
            'value': selectedValue
        }
        filters = [...filters, filter]
    }

    function removeFilter(filter: Filter) {
        filters = filters.filter(f => f != filter);
    }

</script>

<div>
    <select bind:value={selectedObject}>
        <option value=''>Object</option>
        {#each Object.keys(options) as object}
            <option value={object}>{object}</option>
        {/each}
    </select>
    <select bind:value={selectedAttribute}>
        <option value=''>Attribute</option>
        {#if selectedObject}
            {#each options[selectedObject] as attribute}
                <option value={attribute}>{attribute}</option>
            {/each}
        {/if}
    </select>
    <select bind:value={selectedOperator}>
        <option value=''>Operator</option>
        {#if selectedObject && selectedAttribute}
            {#each operators as operator}
                <option value={operator}>{operator}</option>
            {/each}
        {/if}
    </select>
    <input bind:value={selectedValue} />
    <button on:click={() => addFilter()}>Add</button>
    <div>
        {#each filters as filter}
            <div class='filter' on:click={() => removeFilter(filter)}>
                {filter.object}.{filter.attribute} {filter.operator} {filter.value}
            </div>
        {/each}
    </div>
</div>

<style>
    .filter {
        display: inline-block;
        border-radius: 4px;
        padding: 2px;
        margin: 5px;
        background-color: #ddd;
        font-family: 'Roboto-Mono', monospace;
    }
</style>