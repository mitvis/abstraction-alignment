<!-- Svelte component that lists common confusions across the model -->
<script lang='ts'>
    import {onMount} from 'svelte';
    import type { Score, Node, Filter } from './types';
    import * as d3 from 'd3';
    import type { HierarchyNode } from "d3-hierarchy";
    import { isConnected, shortenName, computeEntropy } from './hierarchyUtils';
    import FilterBar from './FilterBar.svelte';

    export let numConfusions: number = 50;
    export let jointEntropy = new Map<string, number>();
    export let hierarchy = [] as Node[];
    export let trees = [] as HierarchyNode<Node>[];
    export let colorMap = new Map<string, string>();
    export let selectedIDs: number[];
    export let numInstances: number = 0;
    $: max2DEntropy = computeEntropy([0.5,0.5]) * numInstances;

    let root: HierarchyNode<Node>;
    let nodes: [string, string][] = [];
    let selectedNodes: [string, string][] = [];
    let idToNode = new Map<string, HierarchyNode<Node>>();
    let treeNodes = [] as Map<string, number>[];

    const maxStringChars = 30;
    
    let filters: Filter[] = [];

    $: if (hierarchy.length > 0 && jointEntropy) {
        hierarchy = JSON.parse(JSON.stringify(hierarchy));
        let rootNode = hierarchy.filter(node => node.parent == null)[0];
        root = d3.hierarchy(rootNode, (d: Node) => 
            hierarchy.filter(node => node.parent === d.id)
        );
        idToNode = new Map(root.descendants().map(node => {
            return [node.data.id.toString(), node]
        }));
        nodes = Array.from(jointEntropy.keys()).map(key => {
            const [key1, key2] = key.split(',');
            return [key1, key2] as [string, string];
        }).sort((a, b) => {
            const aKey = a[0] + ',' + a[1];
            const bKey = b[0] + ',' + b[1];
            return getItem(bKey, jointEntropy) - getItem(aKey, jointEntropy);
        });
        selectedNodes = [...nodes];
        treeNodes = trees.map(root =>
            new Map(root.descendants().map(node => {
                return [node.data.name, node.data.value]
            }))
        );
    }

    function getItem(key: string | number, map: Map<string | number, any>) {
        const item = map.get(key);
        if (!item) { 
            console.log('KEY', key, 'MAP', map, 'KEYS', map.keys(), 'ITEM', item);
            throw new Error('Item not found in map.');
        }
        return item;
    }

    $: if (filters) {
        selectedNodes = nodes.filter((pair, i) => combineFilters()(pair));
    }

    let filterFunctions: {[key: string]: any} = {
        'height': (node: HierarchyNode<Node>) => node.height,
        'depth': (node: HierarchyNode<Node>) => node.depth,
        'name': (node: HierarchyNode<Node>) => node.data.name,
        'connected': connected,
        'shareParent': shareParent,
        'isLabel': isLabel,
    }

    function combineFilters() {
        let combinedFilter = (pair: [string, string]) => {
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

    function connected(pair: [string, string]) {
        const [id1, id2] = pair;
        const node1 = getItem(id1, idToNode);
        const node2 = getItem(id2, idToNode);
        return isConnected(node1, node2);
    }

    function shareParent(pair: [string, string]) {
        const [id1, id2] = pair;
        const node1 = getItem(id1, idToNode);
        const node2 = getItem(id2, idToNode);
        return node1.parent == node2.parent;
    }

    function isLabel(node: HierarchyNode<Node>) {
        return node.parent !== null && !node.data.name.includes('-')
    }

    function applyNodeFilters(node: HierarchyNode<Node>, filters: Filter[]) {
        for (let filter of filters) {
            let functionOutput = filterFunctions[filter.attribute](node);
            let expression = functionOutput + ' ' + filter.operator + ' ' + filter.value;
            if (!evalExpression(expression)) {
                return false;
            }
        }
        return true;
    }

    function applyPairFilters(pair: [string, string], filters: Filter[]) {
        for (let filter of filters) {
            let functionOutput = filterFunctions[filter.attribute](pair);
            let expression = functionOutput + ' ' + filter.operator + ' ' + filter.value;
            if (!eval(expression)) {
                return false;
            }
        }
        return true;
    }

    function instanceFitsPair(pair: [string, string], treeNodes: Map<string, number>[]) {
        const [id1, id2] = pair;
        selectedIDs = treeNodes
            .map((tree, i) => [tree, i])
            .filter(item => {
                return item[0].has(id1) && item[0].has(id2);
            })
            .map(item => item[1]);
    }

    function parseValue(value: string) {
        if (value.toLowerCase() == 'true') { return true};
        if (value.toLowerCase() == 'false') { return false};
        return parseFloat(value);
    }

    function evalExpression(expression: string) {
        let [left, operator, right] = expression.split(' ');

        left = parseValue(left);
        right = parseValue(right);

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

    function parseEntropy(pair: [number, number]) {
        const [id1, id2] = pair;
        const entropy = getItem(id1 + ',' + id2, jointEntropy);
        const percentEntropy = entropy / max2DEntropy;
        return percentEntropy.toFixed(2);
    }

    function getColor(id: number) {
        let node = getItem(id, idToNode);
        if (node.height == 0) {
            node = node.parent
        }
        return colorMap.get(node.data.name);
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
                <div 
                    class='confusion-result'
                    on:click={() => {instanceFitsPair(pair, treeNodes)}}
                >
                    <span 
                        style='background-color:{getColor(pair[0])}'
                        title={getItem(pair[0], idToNode).data.name}
                    >
                        {shortenName(getItem(pair[0], idToNode).data.name, maxStringChars)}
                    </span>
                    <span 
                        style='background-color:{getColor(pair[1])}'
                        title={getItem(pair[1], idToNode).data.name}
                    >
                        {shortenName(getItem(pair[1], idToNode).data.name, maxStringChars)}
                    </span>
                    <span style='width: 2.5em;'>{parseEntropy(pair)}</span>
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
        column-gap: 1px;
    }

    .confusion-result > span {
        display: inline-block;
        padding: 3px;
        background-color: lightgrey;
        margin: 0;
        white-space: nowrap;
        width: 18em;
        font-family: Roboto-Mono, monospace;
        /* font-weight: 700; */
    }

    p {
        font-size: 12pt;
        font-weight: 600;
        margin-bottom: 0;
    }

</style>