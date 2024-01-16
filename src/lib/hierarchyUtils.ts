// Utility functions for working with hierarchies

import type { Tree, Score, Node} from "./types";
import { scaleLinear } from "d3";
import type { HierarchyNode } from "d3-hierarchy";
import * as d3 from 'd3';


/**
 * Assigns scores to nodes in a tree and returns a new tree with the scores.
 *
 * This function uses a linear scale to normalize the scores to a range between
 * 0 and 1, based on the total sum of all scores. If a node's id is found in
 * the scores object, the node's value is set to the normalized score. Nodes
 * that do not have a corresponding score are left unchanged.
 *
 * @param scores - An object where the keys are node ids and the values are the
 * scores for the nodes.
 * @param hierarchy - An array of nodes representing a tree. Each node is an
 * object with an `id` property and optionally a `value` property.
 * 
 * @returns A new tree where each node's value is set to the normalized score
 * if the node's id is found in the scores object. If the scores or hierarchy
 * are not provided, an empty array is returned.
 */
export function assignScores(scores: Score, hierarchy: Node[]) {
    // Assigns scores to the tree and return a new tree
    if (scores && hierarchy) {
        const sumScore = Object.values(scores).reduce((a, b) => a + b, 0);
        const scoreScale = scaleLinear().domain([0, sumScore]).range([0, 1]);
        let tree = hierarchy.map(node => ({...node}));
        tree.forEach(node => {
            if (node.id in scores) {
                node.value = scoreScale(scores[node.id]);
            };
        });
        return tree;
    };
    return [] as Tree;
};

/**
 * Propagates scores through a tree and modifies the tree in place.
 *
 * Each leaf node keeps its existing `value`. All other nodes' values
 * are the sum of the `value` of every reachable leaf node.
 *
 * @param tree - The tree structure to propagate scores through. 
 * An array of nodes, where each node is an object with a `parent` property 
 * and a `value` property. The `parent` property is the index of the parent
 * node in the array, and the `value` property is the score of the node.
 */
export function propagateScores(tree: Node[]) {
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

/**
 * Performs a depth-first traversal of a tree, returning an array of nodes in
 * the order they were visited.
 *
 * @param root - The root node of the tree. The node should have a `height`
 * property and optionally a `children` property, which is an array of child
 * nodes.
 * 
 * @returns An array of nodes in the order they were visited. If the root node
 * is null or its height is 0, an empty array is returned.
 */
export function depthFirstItemization(root: HierarchyNode<Node>): HierarchyNode<Node>[] {
    if (! root || root.height === 0) return [];
    let nodes = [] as HierarchyNode<Node>[];
    let stack = [root];
    while (stack.length > 0) {
        let node = stack.pop();
        if (node) {
            nodes.push(node);
            if (node.children) {
                node.children.forEach(child => stack.push(child));
            }
        };
    }
    return nodes;
};

/**
 * Finds the parent of a node at a given depth in a hierarchy.
 *
 * @param node - The node whose parent is to be found. The node should be an
 * instance of HierarchyNode with a `depth` property and optionally a `parent`
 * property.
 * @param depth - The depth at which to find the parent.
 * 
 * @returns The parent node at the specified depth. If no parent is found at the
 * given depth, an error is thrown.
 * @throws {Error} Throws an error if no parent is found at the given depth.
 */
export function findParent(node: HierarchyNode<Node>, depth: number) {
    // Finds the parent of the node at the given depth
    if (node.depth === depth) return node;
    if (node.parent) return findParent(node.parent, depth);
    
    // Raise an error if we get here
    throw new Error(`Could not find parent of node ${node} at depth ${depth}`);
};

/**
 * Checks if a child node is connected to a specified parent node in a hierarchy.
 *
 * @param child - The child node to check. The node should be an instance of
 * HierarchyNode with a `parent` property.
 * @param parent - The parent node to check against.
 * 
 * @returns True if the child node is connected to the parent node, false
 * otherwise.
 */
export function isAncestor(child: HierarchyNode<Node>, parent: HierarchyNode<Node>) {
    // Checks whether the child is connected to the parent
    if (child.parent === parent) return true;
    if (child.parent) return isAncestor(child.parent, parent);
    return false;
};

/**
 * Checks whether two nodes in a hierarchy are connected to each other.
 * A node is considered connected to another node if it is an ancestor of the
 * other node.
 *
 * @param {HierarchyNode<Node>} nodeA - One node to check for a connection.
 * @param {HierarchyNode<Node>} nodeB - Another node to check for a connection.
 * 
 * @returns {boolean} - Returns true if nodeA is an ancestor of nodeB or nodeB
 * is an ancestor of nodeA. Otherwise, it returns false.
 */
export function isConnected(nodeA: HierarchyNode<Node>, nodeB: HierarchyNode<Node>) {
    // Checks whether the nodes are connected to each other
    if (nodeA.depth > nodeB.depth) {
        return isAncestor(nodeA, nodeB);
    } else {
        return isAncestor(nodeB, nodeA);
    };
}

/**
 * Computes the entropy of a discrete probability distribution.
 *
 * @param probabilities - An array of numbers representing the probabilities of 
 * each outcome in a discrete distribution. Each number should be between 0 and 1, 
 * and the sum of all numbers should be 1. If not, the array will be normalized.
 *
 * @returns The entropy of the distribution, as a number.
 */
export function computeEntropy(values: number[]): number {
    // Calculate the sum of all probabilities
    let sum = values.reduce((a, b) => a + b, 0);

    // Normalize probabilities so they are between 0 and 1 and they sum to 1
    let probabilities = values.map(value => value / sum);

    let entropy = probabilities.reduce((entropy, probability) => {
        if (probability > 0) {
            return entropy - probability * Math.log(probability);
        } else {
            return entropy;
        }
    }, 0);
    return entropy;
};

/**
 * Creates a hierarchy tree from the given nodes and scores, and extracts the
 * root.
 *
 * @param {Node[]} hierarchy - Nodes to create the hierarchy from.
 * @param {Score} scores - Scores to assign to the nodes.
 * @param {number} threshold - Threshold to use when extracting the root.
 * @returns {HierarchyNode<Node>} - The root of the hierarchy tree.
 */
export function createHierarchy(hierarchy: Node[], scores: Score, threshold: number) {
    const tree = assignScores(scores, hierarchy);
    propagateScores(tree);
    return extractD3HierarchyRoot(tree, threshold);
}

/**
 * Extracts the root of a hierarchy tree from the given nodes.
 *
 * @param {Node[]} hierarchy - Nodes to extract the root from.
 * @param {number | null} threshold - Threshold to use when extracting the root.
 * @returns {HierarchyNode<Node>} - The root of the hierarchy tree.
 */
export function extractD3HierarchyRoot(hierarchy: Node[], threshold: number | null = null) {
    const rootNode = hierarchy.filter(node => node.parent == null)[0];
    const root = d3.hierarchy(rootNode, (d: Node) => 
        hierarchy.filter(node => node.parent === d.id && (threshold === null || (node.value !== null && node.value >= threshold)))
    ).sort((a: HierarchyNode<Node>, b: HierarchyNode<Node>) => 
        (a.data.value ?? 0) - (b.data.value ?? 0)
    );
    return root;
}

/**
 * Gets all nodes at a specific depth in a hierarchy tree.
 *
 * @param {HierarchyNode<Node>} root - The root of the hierarchy tree.
 * @param {number} depth - Depth to get the nodes from.
 * @returns {HierarchyNode<Node>[]} - The nodes at the specified depth.
 */
export function getNodesAtDepth(root: HierarchyNode<Node>, depth: number) {
    const nodes = depthFirstItemization(root);
    return nodes.filter(node => node.depth === depth);
}
