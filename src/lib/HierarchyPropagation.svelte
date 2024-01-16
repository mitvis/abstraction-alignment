<!-- A Svelte component that visualizes the an uncertainty fingerprint -->
<script lang='ts'>
    import type { Node, Tree, Score, VisualizationAttributes } from "./types.ts";
    import * as d3 from 'd3';
    import type { HierarchyNode } from "d3-hierarchy";
    import { createEventDispatcher, onMount } from "svelte";
    import { depthFirstItemization, isAncestor, findParent } from "./hierarchyUtils";
    import { schemeSet1, schemeSet3 } from 'd3-scale-chromatic';

    export let hierarchy: Node[];
    export let scores: Score;
    export let root: HierarchyNode<Node>;
    export let levelIDMap: Map<number, number>;
    export let width = 500; // width of the SVG
    export let maxStringChars = 18;
    export let collapseDepth = 3;
    
    const dispatch = createEventDispatcher();
    const colorPalette = [...schemeSet1.slice(0, -1), ...schemeSet3];

    let selectedRoot: HierarchyNode<Node>;
    let visualizationAttributes: VisualizationAttributes = {};
    $: numVisibleNodes = Object.values(visualizationAttributes).filter(attribute => attribute.isVisible).length;
    
    // When data is loaded, create a hierarchy and dispatch a loaded event
    $: if (hierarchy && scores) {
        selectedRoot = root;
        visualizationAttributes = initializeVisualizationAttributes(root);
        dispatch('loaded');
    }

    onMount(() => {
        dispatch('loaded');
    });

    // Create a mapping to color each node
    let colorMap = new Map<number, string>();
    // let colorIDMap = new Map<number, number>(); // maps node IDs to their color index (depth order)

    $: if (hierarchy) {
        colorMap = hierarchy.reduce((acc: Map<number, string>, node: Node) => 
            acc.set(node.id, colorPalette[levelIDMap.get(node.id)! % colorPalette.length]), new Map<number, string>);
    }

    // Select the nodes to color based on the selected root
    let colorNodes: {[key: number]: [number, HierarchyNode<Node>]} = {};
    $: if (selectedRoot && selectedRoot.children) {
        colorNodes = selectedRoot.children
            .reduce((acc: {}, node: HierarchyNode<Node>, index: number) => ({...acc, [node.data.id]:[index, node]}), {})
    }

    // Dynamically update the size of the SVG
    const padding = 5; // svg padding
    const indent = 10; // indent for each level of the tree
    const itemHeight = 15; // height of a row item
    const lineHeight = 5; // height of the background line
    const iconSize = 7; // height/width of the collapse icon
    $: height = (numVisibleNodes * itemHeight) + (padding * 2);

    const iconX = (node: HierarchyNode<Node>) => padding + (node.depth * indent);
    const textX = (node: HierarchyNode<Node>) => iconSize + padding + iconX(node);
    $: textY = (node: HierarchyNode<Node>) => padding + (visualizationAttributes[node.data.id].yIndex * itemHeight) + (itemHeight / 2);
    $: lineY = (node: HierarchyNode<Node>) => textY(node) - (lineHeight);
    const baseLineWidth = (node: HierarchyNode<Node> | null) => node && node.parent ? scaleLine(node.parent.data.value) : scaleLine(1);
    const baseLineX = (node: HierarchyNode<Node> | null) => node && node.parent ? lineOffsetX + scaleLine(visualizationAttributes[node.parent.data.id].x0) : lineOffsetX;
    const sparkLineX = (node: HierarchyNode<Node>) => lineOffsetX + scaleLine(visualizationAttributes[node.data.id].x0);

    $: labelWidth = maxStringChars * 8.6;
    $: confidenceWidth = root ? 
        root.descendants()
            .reduce((max, node) => node.data.value!.toFixed(2).toString().length > max 
                ? node.data.value!.toFixed(2).toString().length 
                : max, 
            0) * 10
        : 30;
    $: totalLineWidth = width - labelWidth - confidenceWidth - (2 * padding);
    $: scaleLine = d3.scaleLinear().domain([0, 1]).range([0, totalLineWidth]);
    $: lineOffsetX = 7 * padding + labelWidth;


    function shortenName(name: string, maxChars: number = (maxStringChars - 4)) {
        name = name.replace('_', ' ');
        return name.length > maxChars ? name.substring(0, maxChars) + '...' : name;
    }

    /** Creates the visualization attributes for each node */
    function initializeVisualizationAttributes(root: HierarchyNode<Node>) {
        let visAttributes = {} as VisualizationAttributes;
        if ( ! root || !root.data ) return visAttributes;
        let stack = [root];
        let yIndex = 0;
        let depthX0 = new Array(root.height + 1).fill(0);
        let priorDepth = 0;
        let priorX0 = 0;
        while (stack.length > 0) {
            let node = stack.pop();
            if (! node) continue;

            // Set the initial collapse state
            let collapsed = node.depth > collapseDepth;
            let visible = node.depth <= collapseDepth;

            // Moving down the tree, start from parent's x0
            if (node.depth > priorDepth) {
                depthX0[node.depth] = priorX0;
            ;}
            
            // Create attribute
            let attribute = {
                id: node.data.id,
                name: node.data.name,
                isCollapsed: collapsed, 
                isVisible: visible, 
                yIndex: yIndex, 
                x0: depthX0[node.depth]
            };
            visAttributes[node.data.id] = attribute;

            // Moving up the tree, move children forward to make up any gaps (from probabilities not summing to 1)
            if (node.depth < priorDepth) {
                depthX0[priorDepth] = depthX0[node.depth];
            };

            // Moving to a sibling, move all children forward
            if (node.depth == priorDepth) {
                depthX0.fill(depthX0[node.depth], node.depth + 1);
            };

            // Update the parameters
            depthX0[node.depth] += node.data.value;
            priorDepth = node.depth;
            priorX0 = attribute.x0;

            // If the node is visible, then move down one row for the next node.
            if (visible) { yIndex += 1;}
            
            // Add the children to the stack to continue the traversal
            if (node.children) {
                node.children.forEach(child => stack.push(child));
            }
            
        }
        return visAttributes;
    }

    /** Recursively sets the visibility of the node and its children */
    function recursiveVisibility(node: HierarchyNode<Node>, visible=true) {
        visualizationAttributes[node.data.id].isVisible = visible;
        let collapsed = visualizationAttributes[node.data.id].isCollapsed;
        let childVisible = visible && !collapsed;
        if (node.children) {
            node.children.forEach(child => recursiveVisibility(child, childVisible));
        }
    }

    /** Toggles the collapse state of the node */
    function toggleCollapse(node: HierarchyNode<Node>) {
        // if node is collapsed, then expand it
        let collapse = ! visualizationAttributes[node.data.id].isCollapsed;
        visualizationAttributes[node.data.id].isCollapsed = collapse;
        visualizationAttributes[node.data.id].isVisible = true;

        // make its children visible if it is expanded
        if (node.children) {
            node.children.forEach(child => recursiveVisibility(child, !collapse));
        }
    }

    /** Updates the yIndex of each node based on its visibility */
    function reindex() {
        let currentYIndex = 0;
        let orderedNodes = depthFirstItemization(root);
        for (let i = 0; i < orderedNodes.length; i++) {
            visualizationAttributes[orderedNodes[i].data.id].yIndex = currentYIndex;
            if ( visualizationAttributes[orderedNodes[i].data.id].isVisible) {
                currentYIndex += 1;
            }
        }
    }

    /** Returns the color of the node */
    function color(node: HierarchyNode<Node>, root: HierarchyNode<Node>) {
        if (node.depth <= root.depth) return 'darkgrey';
        if (! isAncestor(node, root)) return 'darkgrey';
        if (node && node.depth > root.depth + 1) {
            node = findParent(node, root.depth + 1);
        }
        const color = colorMap.get(node.data.id);
        return color;
    }


</script>

<div class='hierarchy' style="gap: {padding}px">
    {#if root}
        <svg width={width} height={height}>
            {#each depthFirstItemization(root) as node}
                {#if visualizationAttributes[node.data.id].isVisible}
                    <g style='display: flex; flex-direction: row' >
                        {#if node.children}
                            {#if visualizationAttributes[node.data.id].isCollapsed} // upward pointing triangle
                                <polygon points="{iconX(node)},{textY(node)} {iconX(node) + iconSize / 2},{textY(node) - iconSize} {iconX(node) + iconSize},{textY(node)}"/>
                            {:else} // downward pointing triangle
                                <polygon points="{iconX(node)},{textY(node) - iconSize} {iconX(node) + iconSize / 2},{textY(node)} {iconX(node) + iconSize},{textY(node) - iconSize}"/>
                            {/if}
                        {/if}
                        <text x={textX(node)} y={textY(node)} font-family='Roboto Mono, monospace'>
                            {shortenName(node.data.name)}
                        </text>
                        <text x={labelWidth} y={textY(node)} font-family='Roboto Mono, monospace'>
                            {node.data.value ? node.data.value.toFixed(2).toString() : ''}
                        </text>
                        <rect x={baseLineX(node)} y={lineY(node)} width={baseLineWidth(node)} height={lineHeight} fill='#e3e3e3' />
                        {#if node != selectedRoot}
                            <rect x={sparkLineX(node)} y={lineY(node)} width={scaleLine(node.data.value)} height={lineHeight} fill={color(node, selectedRoot)} />
                        {/if}
                        {#if node == selectedRoot}
                            {#each Object.entries(colorNodes) as [_, nodeInfo]}
                                <rect x={sparkLineX(nodeInfo[1])} y={lineY(node)} width={scaleLine(nodeInfo[1].data.value)} height={lineHeight} fill={color(nodeInfo[1], selectedRoot)} />
                            {/each}
                        {/if}
                        <rect 
                            x={0} 
                            y={textY(node) - itemHeight} 
                            width={labelWidth} 
                            height={itemHeight} 
                            fill='transparent'
                            role='button'
                            tabindex="0"
                            on:click={async () => {await toggleCollapse(node); await reindex();}} 
                            on:keypress={async () => {await toggleCollapse(node); await reindex();}}
                        >
                            <title>{node.data.name}</title>
                        </rect>
                        <rect
                            x={lineOffsetX} 
                            y={textY(node) - itemHeight} 
                            width={totalLineWidth} 
                            height={itemHeight} 
                            fill='transparent'
                            role='button'
                            tabindex="0"
                            on:click={() => {selectedRoot = node;}} 
                            on:keypress={() => {selectedRoot = node;}}
                        />
                    </g>
                {/if}
            {/each}
        </svg>
    {/if}
</div>

<style>
    .hierarchy {
        display: flex; 
        flex-direction: row;
    }
    rect:focus {
        outline: none;
    }
</style>