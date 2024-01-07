<!-- A Svelte component that visualizes the an uncertainty fingerprint -->
<script lang='ts'>
    import type { Node, Tree, Score, VisualizationAttributes } from "./types.ts";
    import * as d3 from 'd3';
    import type { HierarchyNode } from "d3-hierarchy";

    const colorScheme = d3.scaleOrdinal(d3.schemeSet1);

    export let instanceID = 0;
    export let datasetPath = "/data/cifar/";

    let hierarchy = [] as Tree;
    let scores = {} as Score;

    function loadData() {
        d3.json(`${datasetPath}/hierarchy.json`).then((jsonObject: Tree) => {
            hierarchy = jsonObject;
        });
        d3.json(`${datasetPath}/scores.json`).then((jsonObject: {[key: number]: Score}) => {
            scores = jsonObject[instanceID];
        });
    }

    $: loadData();

    // An uncertainty fingerprint is defined as a Tree and Scores
    // export let scores = {}; // maps IDs to scores
    // export let hierarchy = [] as Tree; // basic hierarchy structure (does not change)
    export let threshold: number = 0.01; // threshold for filtering nodes
    export let width: number = 500; // width of the SVG
    
    // Create a tree by propagating scores through the hierarchy and filtering
    let tree = [] as Tree; // scores propagated through the hierarchy
    $: tree = assignScores(scores, hierarchy);
    $: propagateScores(tree);

    // Create a d3 hierarchy object for easy filtering
    $: rootNode = tree ? tree.filter(node => node.parent == null)[0]: {};
    $: root = d3.hierarchy(rootNode, (d: Node) => 
        tree ? tree.filter(node => node.parent === d.id && (node.value != null && node.value > threshold)) : []
    );
    $: root.sort((a: HierarchyNode<Node>, b: HierarchyNode<Node>) => 
        (a.data.value ?? 0) - (b.data.value ?? 0)
    );
    $: selectedRoot = root;
    let colorNodes: { [key: number] : [number, HierarchyNode<Node>]} = {};
    $: colorNodes = selectedRoot.children ? selectedRoot.children.reduce((acc: Object, node: HierarchyNode<Node>, index: number) => ({...acc, [node.data.id]:[index, node]}), {}) : {};

    // Order the nodes in the print order
    let orderedNodes = [] as HierarchyNode<Node>[];
    $: orderedNodes = depthFirstItemization(root);

    // Store visualization attributes for each node
    let visualizationAttributes = {} as VisualizationAttributes;
    $: visualizationAttributes = initializeVisualizationAttributes(root);

    // Dynamically update the size of the SVG
    const padding = 5; // svg padding
    const indent = 10; // indent for each level of the tree
    const itemHeight = 15; // height of a row item
    const baseLineHeight = 5; // height of the background line
    const sparkLineHeight = 5; // height of the spark line
    const iconSize = 7; // height/width of the collapse icon
    $: labelWidth = tree ? tree.reduce((max, node) => shortenName(node.name).length > max ? shortenName(node.name).length : max, 0) * 9 : 200;
    $: confidenceWidth = tree ? tree.reduce((max, node) => node.value.toFixed(2).toString().length > max ? node.value.toFixed(2).toString().length : max, 0) * 10 : 30;
    $: totalLineWidth = width - labelWidth - confidenceWidth - (2 * padding);
    $: height = (orderedNodes.length * itemHeight) + (padding * 2);
    $: scaleLine = d3.scaleLinear().domain([0, 1]).range([0, totalLineWidth]);
    $: iconX = (node: HierarchyNode<Node>) => padding + (node.depth * indent);
    $: textX = (node: HierarchyNode<Node>) => iconSize + padding + iconX(node);
    $: textY = (node: HierarchyNode<Node> | null) => {
        if (node) {
            return padding + visualizationAttributes[node.data.id].yIndex * itemHeight + (itemHeight / 2);
        }
        return padding + (itemHeight / 2);
    }
    $: confidenceX = (node: HierarchyNode<Node>) => labelWidth;
    $: lineY = (node: HierarchyNode<Node> | null) => textY(node) - (sparkLineHeight);
    $: lineOffsetX = 7 * padding + labelWidth;
    $: baseLineWidth = (node: HierarchyNode<Node> | null) => node && node.parent ? scaleLine(node.parent.data.value) : scaleLine(1);
    $: baseLineX = (node: HierarchyNode<Node> | null) => node && node.parent ? lineOffsetX + scaleLine(visualizationAttributes[node.parent.data.id].x0) : lineOffsetX;
    $: sparkLineX = (node: HierarchyNode<Node>) => lineOffsetX + scaleLine(visualizationAttributes[node.data.id].x0);

    function shortenName(name: string, maxChars=15) { 
        if (name.length > maxChars) {
            return name.substring(0, maxChars) + '...';
        };
        return name;
    }
    

    function assignScores(scores: Score, hierarchy: Tree) {
        // Assigns scores to the tree and return a new tree
        if (scores && hierarchy) {
            const sumScore = Object.values(scores).reduce((a, b) => a + b, 0);
            const scoreScale = d3.scaleLinear().domain([0, sumScore]).range([0, 1]);
            tree = hierarchy.map(node => ({...node}));
            tree.forEach(node => {
                if (node.id in scores) {
                    node.value = scoreScale(scores[node.id]);
                };
            });
            return tree;
        };
        return [] as Tree;
    };

    function propagateScores(tree: Tree) {
        // Propagates scores through the tree
        if (tree.length == 0) return;
        const root = tree.filter(node => node.parent == null)[0];
        if ( ! root.value || root.value === 0) {
            // Scores are not propagated, so sum the leaf values through all parent values
            tree.forEach(node => node.value ? node.value = node.value : node.value = 0);
            tree.forEach(node => {
                if (node.value && node.value > 0) {
                    let current = node;
                    while (current.parent != null) {
                        if (tree[current.parent].value == null) {
                            tree[current.parent].value = 0;
                        };
                        tree[current.parent].value += node.value;
                        current = tree[current.parent];
                    };
                };
            });
        };
    };

    function depthFirstItemization(root: HierarchyNode<Node>) {
        // Returns a list of nodes in depth first order.
        if (! root || root.height === 0) return [];
        let nodes = [];
        let stack = [root];
        while (stack.length > 0) {
            let node = stack.pop();
            nodes.push(node);
            if (node && node.children) {
                node.children.forEach(child => stack.push(child));
            }
        }
        return nodes;
    }

    function initializeVisualizationAttributes(root: HierarchyNode<Node>) {
        // Initialize the visualization attributes for each node
        let visualizationAttributes = {} as VisualizationAttributes;
        if ( ! root || !root.data ) return visualizationAttributes;
        let stack = [root];
        let yIndex = 0;
        let depthX0 = new Array(root.height + 1).fill(0);
        let priorDepth = 0;
        let priorX0 = 0;
        while (stack.length > 0) {
            let node = stack.pop();
            if (! node) continue;
            // Set the initial collapse state
            let collapsed = node.depth > 3;
            let visible = node.depth <= 3;

            // Moving down the tree, start from parent's x0
            if (node.depth > priorDepth) {
                depthX0[node.depth] = priorX0;
            ;}
            
            // create attribute
            let attribute = {
                id: node.data.id,
                name: node.data.name,
                isCollapsed: collapsed, 
                isVisible: visible, 
                yIndex: yIndex, 
                x0: depthX0[node.depth]
            };
            visualizationAttributes[node.data.id] = attribute;

            // Moving up the tree, move children forward to make up any gaps (from probabilities not summing to 1)
            if (node.depth < priorDepth) { // moving up a level in the tree
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
            
            // Add the chidren to the stack to continue the traversal
            if (node.children) {
                node.children.forEach(child => stack.push(child));
            }
            
        }
        return visualizationAttributes;
    }

    function recursiveVisibility(node: HierarchyNode<Node>, visible=true) {
        visualizationAttributes[node.data.id].isVisible = visible;
        let collapsed = visualizationAttributes[node.data.id].isCollapsed;
        let childVisible = visible && !collapsed;
        if (node.children) {
            node.children.forEach(child => recursiveVisibility(child, childVisible));
        }
    }

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

    function reindex() {
        let currentYIndex = 0;
        for (let i = 0; i < orderedNodes.length; i++) {
            visualizationAttributes[orderedNodes[i].data.id].yIndex = currentYIndex;
            if ( visualizationAttributes[orderedNodes[i].data.id].isVisible) {
                currentYIndex += 1;
            }
        }
    }

    function findParent(node: HierarchyNode<Node>, depth: number) {
        if (node.depth === depth) return node;
        if (node.parent) return findParent(node.parent, depth);
        return node; // TODO: switch this to throw an error
    }

    function isConnected(child: HierarchyNode<Node>, parent: HierarchyNode<Node>) {
        if (child.parent === parent) return true;
        if (child.parent) return isConnected(child.parent, parent);
        return false;
    }

    function color(node: HierarchyNode<Node>, root: HierarchyNode<Node>) {
        if (node.depth <= root.depth) return 'darkgrey';
        if (! isConnected(node, root)) return 'darkgrey';
        if (node && node.depth > root.depth + 1) {
            node = findParent(node, root.depth + 1);
        }
        const color = colorScheme(node.data.id);
        return color;
    }

</script>

<div class='hierarchy' style="gap: {padding}px">
    {#if root.data}
        <svg width={width} height={height}>
            {#each orderedNodes as node}
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
                        <text x={confidenceX(node)} y={textY(node)} font-family='Roboto Mono, monospace'>
                            {node.data.value ? node.data.value.toFixed(2).toString() : ''}
                        </text>
                        <rect x={baseLineX(node)} y={lineY(node)} width={baseLineWidth(node)} height={baseLineHeight} fill='lightgrey' />
                        {#if node != selectedRoot}
                            <rect x={sparkLineX(node)} y={lineY(node)} width={scaleLine(node.data.value)} height={sparkLineHeight} fill={color(node, selectedRoot)} />
                        {/if}
                        {#if node == selectedRoot}
                            {#each Object.entries(colorNodes) as [_, nodeInfo]}
                                <rect x={sparkLineX(nodeInfo[1])} y={lineY(node)} width={scaleLine(nodeInfo[1].data.value)} height={sparkLineHeight} fill={color(nodeInfo[1], selectedRoot)} />
                            {/each}
                        {/if}
                        <rect x={0} y={textY(node) - itemHeight} width={labelWidth} height={itemHeight} fill='transparent' on:click={async () => {await toggleCollapse(node); await reindex();}} on:keypress={async () => {await toggleCollapse(node); await reindex();}}/>
                        <rect x={lineOffsetX} y={textY(node) - itemHeight} width={totalLineWidth} height={itemHeight} fill='transparent' on:click={() => {selectedRoot = node;}} on:keypress={() => {selectedRoot = node;}}/>
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
</style>