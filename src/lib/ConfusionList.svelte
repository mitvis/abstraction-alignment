<!-- Svelte component that lists common confusions across the model -->
<script lang='ts'>
    import {onMount} from 'svelte';
    import type { Score, Node, Filter } from './types';
    import * as d3 from 'd3';
    import type { HierarchyNode } from "d3-hierarchy";
    import { isConnected, shortenName } from './hierarchyUtils';
    import FilterBar from './FilterBar.svelte';

    export let numConfusions: number = 50;
    export let jointEntropy = new Map<string, number>();
    export let hierarchy = [] as Node[];
    export let colorMap = new Map<number, string>();

    let root: HierarchyNode<Node>;
    let nodes: [number, number][] = [];
    let selectedNodes: [number, number][] = [];
    let idToNode = new Map<number, HierarchyNode<Node>>();
    let maxStringChars = 13;
    
    let filters: Filter[] = [];

    $: if (hierarchy.length > 0 && jointEntropy) {
        hierarchy = JSON.parse(JSON.stringify(hierarchy));
        let rootNode = hierarchy.filter(node => node.parent == null)[0];
        root = d3.hierarchy(rootNode, (d: Node) => 
            hierarchy.filter(node => node.parent === d.id)
        );
        idToNode = new Map(root.descendants().map(node => {
            return [node.data.id, node]
        }));
        nodes = Array.from(jointEntropy.keys()).map(key => {
            const [key1, key2] = key.split(',');
            return [parseInt(key1), parseInt(key2)] as [number, number];
        }).sort((a, b) => {
            const aKey = a[0] + ',' + a[1];
            const bKey = b[0] + ',' + b[1];
            return getItem(bKey, jointEntropy) - getItem(aKey, jointEntropy);
        });
        selectedNodes = [...nodes];
    }

    function getItem(key: string | number, map: Map<string | number, any>) {
        const item = map.get(key);
        if (!item) { throw new Error('Item not found in map.');}
        return item;
    }

    $: if (filters) {
        selectedNodes = nodes.filter(pair => combineFilters()(pair));
    }

    let filterFunctions = {
        'connected': connected,
        'shareParent': shareParent
    }

    function combineFilters() {
        let combinedFilter = (pair: [number, number]) => {
            const [id1, id2] = pair;
            const node1 = getItem(id1, idToNode);
            const node2 = getItem(id2, idToNode);

            const nodeAFilters = filters.filter(filter => filter.object == 'nodeA');
            const nodeBFilters = filters.filter(filter => filter.object == 'nodeB');
            const nodeFiltersFitA = applyNodeFilters(node1, nodeAFilters) && applyNodeFilters(node2, nodeBFilters)
            const nodeFiltersFitB = applyNodeFilters(node2, nodeAFilters) && applyNodeFilters(node1, nodeBFilters)
            if (!(nodeFiltersFitA || nodeFiltersFitB)) {
                return false;
            }

            const pairFilters = filters.filter(filter => filter.object == 'pair');
            if (!applyPairFilters(pair, pairFilters)) {
                return false;
            }
            return true;
        }
        return combinedFilter
    }

    function connected(pair: [number, number]) {
        const [id1, id2] = pair;
        const node1 = getItem(id1, idToNode);
        const node2 = getItem(id2, idToNode);
        return isConnected(node1, node2);
    }

    function shareParent(pair: [number, number]) {
        const [id1, id2] = pair;
        const node1 = getItem(id1, idToNode);
        const node2 = getItem(id2, idToNode);
        return node1.parent == node2.parent;
    }

    function applyNodeFilters(node: HierarchyNode<Node>, filters: Filter[]) {
        for (let filter of filters) {
            let expression = node[filter.attribute] + ' ' + filter.operator + ' ' + filter.value;
            if (!evalExpression(expression)) {
                return false;
            }
        }
        return true;
    }

    function applyPairFilters(pair: [number, number], filters: Filter[]) {
        for (let filter of filters) {
            let functionOutput = filterFunctions[filter.attribute](pair);
            let expression = functionOutput + ' ' + filter.operator + ' ' + filter.value;
            if (!eval(expression)) {
                return false;
            }
        }
        return true;
    }


    function evalExpression(expression) {
        let [left, operator, right] = expression.split(' ');

        left = parseFloat(left);
        right = parseFloat(right);

        switch (operator) {
            case '>':
                return left > right;
            case '>=':
                return left >= right;
            case '<':
                return left < right;
            case '<=':
                return left <= right;
            case '==':
                return left == right;
            case '!=':
                return left != right;
            default:
                throw new Error(`Unknown operator: ${operator}`);
        }
    }

</script>

<div id='confusions'>
    <div id='confusion-filter'>
        <p>Global Filter: </p>
        <FilterBar bind:filters={filters}/>
    </div>
    
    <div class='confusion-results'>
        {#if selectedNodes.length > 0}
            {#each selectedNodes.slice(0, numConfusions) as pair}
                <div class='confusion-result'>
                    <span style='background-color:{colorMap.get(pair[0])}'>
                        {shortenName(getItem(pair[0], idToNode).data.name, maxStringChars)}
                    </span>
                    <span style='background-color:{colorMap.get(pair[1])}'>
                        {shortenName(getItem(pair[1], idToNode).data.name, maxStringChars)}
                    </span>
                    <span style='width: 2.5em;'>{getItem(pair[0] + ',' + pair[1], jointEntropy).toFixed(0)}</span>
                </div>
            {/each}
        {/if}
    </div> 
    
</div> 

<style>

    #confusions {
        display: flex;
        flex-direction: column;
        column-gap: 1em;
        height: 100%;
        row-gap: 1em;
    }

    #confusion-filter {
        display: flex;
        flex-direction: row;
        align-items: flex-end;
        column-gap: 1em;
    }

    .confusion-results {
        display: flex;
        flex-direction: column;
        row-gap: 1em;
        overflow-y: scroll;
    }

    .confusion-result {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }

    .confusion-result > span {
        display: inline-block;
        padding: 3px;
        background-color: lightgrey;
        margin: 0;
        white-space: nowrap;
        width: 8em;
        font-family: Roboto-Mono, monospace;
        /* font-weight: 700; */
    }

    p {
        font-size: 12pt;
        font-weight: 600;
        margin-bottom: 0;
    }

</style>