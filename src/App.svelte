<script lang="ts">
    import {json} from "d3";
    import InstanceList from "./lib/InstanceList.svelte";
    import Filter from "./lib/Filter.svelte";
    import ProjectionPlot from "./lib/ProjectionPlot.svelte";
    import QueryBar from "./lib/QueryBar.svelte";
    import type { Embedding, Node, Score } from "./lib/types";
    import ConfusionList from "./lib/ConfusionList.svelte";
    import {onMount} from "svelte";
    import SearchBar from "./lib/SearchBar.svelte";
    import type { HierarchyNode } from "d3-hierarchy";
    import { createHierarchy, extractD3HierarchyRoot } from "./lib/hierarchyUtils";
    import { schemeSet1, schemeSet3 } from 'd3-scale-chromatic';

    type Dataset = {
        name: string,
        type: string | null,
        threshold: number,
        useLabels: boolean
    };

    const datasets: Dataset[] = [
        {name: 'cifar', type: 'image', threshold: 0.01, useLabels: false},
        {name: 'mimic', type: null, threshold: 0.1, useLabels: true},
    ];
    let dataset = datasets[0];

    const colorPalette = [...schemeSet1.slice(0, -1), ...schemeSet3];
    
    let datasetLabels = [] as string[][];
    let hierarchy = [] as Node[];
    let scores = [] as Score[];
    let predictions = [] as string[][];
    let jointEntropy = new Map<string, number>();
    let trees = [] as HierarchyNode<Node>[];
    let colorMap = new Map<string, string>();
    let numCorrect: number;

    let selectedIDs = [] as number[];
    let instances = [] as string[];
    let instanceLabels = [] as string[][];

    

    $: selectedNumCorrect = selectedIDs.filter(id => isCorrect(predictions[id], datasetLabels[id])).length;

    function loadDataset(dataset: Dataset) {
        let labelsPromise = json(`/data/${dataset.name}/labels.json`);
        let hierarchyPromise = json(`/data/${dataset.name}/hierarchy.json`);
        let scoresPromise = json(`/data/${dataset.name}/scores.json`);
        let jointEntropyPromise = json(`/data/${dataset.name}/joint_entropy.json`);
        if (dataset.useLabels) {
            jointEntropyPromise = json(`/data/${dataset.name}/joint_entropy_labels.json`);
        }
        Promise.all([labelsPromise, hierarchyPromise, scoresPromise, jointEntropyPromise])
        .then(([labelsJson, hierarchyJson, scoresJson, jointEntropyJson]) => {
            console.log('LOADING DATA...')
            datasetLabels = labelsJson as string[][];
            selectedIDs = Array.from({length: datasetLabels.length}, (_, i) => i);
            hierarchy = hierarchyJson as Node[];
            let nameToID = new Map(hierarchy.map((node) => [node.name, node.id]));
            scores = scoresJson as Score[];
            if (dataset.useLabels) {
                scores = datasetLabels.map((labels) => {
                    let score = {} as Score;
                    labels.forEach((label) => {score[nameToID.get(label)!] = 1;});
                    return score;
                });
            }
            predictions = scores
                .map((score: Score) => [
                    Array.from(Object.entries(score))
                        .reduce((scoreA, scoreB) => scoreB[1] > scoreA[1] ? scoreB : scoreA)[0]
                ]);
            numCorrect = predictions.filter((prediction, index) => isCorrect(prediction, datasetLabels[index])).length;
            jointEntropy = new Map(Object.entries(jointEntropyJson));
            trees = scores.map(score => createHierarchy(hierarchy, score, dataset.threshold));
            let hierarchyD3: HierarchyNode<Node>[] = extractD3HierarchyRoot(hierarchy).descendants();
            hierarchyD3.sort((a, b) => a.data.name.localeCompare(b.data.name)).sort((a, b) => a.depth - b.depth);
            let levelIDMap = new Map(hierarchyD3.map((node, index) => [node.data.name, index]));
            colorMap = hierarchy.reduce((acc: Map<string, string>, node: Node) => 
                acc.set(node.name, colorPalette[levelIDMap.get(node.name)! % colorPalette.length]), new Map<string, string>);
            console.log('...DATA LOADING COMPLETE')
        });
    };

    function isCorrect(prediction: string[], label: string[]) {
        return prediction.length === label.length && prediction.sort().every((value, index) => value === label.sort()[index]);
    }

    function loadInstances(selectedIDs: number[], dataset: Dataset) {
        if (dataset.type == 'image') {
            instances = selectedIDs.map(id => getImageURL(id, dataset));
        } else {
            instances = selectedIDs.map(id => id.toString());
        };
        instanceLabels = selectedIDs.map(id => datasetLabels ? datasetLabels[id] : []);
    }

    function getImageURL(instanceID: number, dataset: Dataset) {
        const index = instanceID.toString().padStart(5, "0");
        return `/data/${dataset.name}/images/${dataset.name}_test_${index}.jpg`
    }

    $: loadDataset(dataset);
    $: loadInstances(selectedIDs, dataset);

</script>

<main>
    {#if trees.length > 0}
    <div id='main'>
        <div id='dataset'>
        <p>Dataset: </p>
        {#each datasets as d}
            <button class:active={dataset.name == d.name} on:click={() => dataset = d}>
            {d.name}
            </button>
        {/each}
        </div>
        <div id='content'>
            <div id=instances>
                <p>
                    Selected {selectedIDs.length} of {datasetLabels.length} instances 
                    ({(selectedIDs.length / datasetLabels.length * 100).toFixed(1)}% total 
                    {(selectedNumCorrect ? (selectedNumCorrect / numCorrect * 100) : 0).toFixed(1)}% correct
                    {((selectedIDs.length - selectedNumCorrect) / (datasetLabels.length - numCorrect) * 100).toFixed(1)}% incorrect)
                </p>
                <InstanceList
                    instanceIDs={selectedIDs} 
                    instanceLabels={instanceLabels} 
                    instances={instances} 
                    instanceType={dataset.type}
                    hierarchy={hierarchy}
                    scores={scores}
                    trees={trees}
                    colorMap={colorMap}
                />
            </div>
            <div id='right-panel'>
                <div id='controls'>
                    <SearchBar bind:selectedIDs={selectedIDs}/>
                    <QueryBar trees={trees} bind:selectedIDs={selectedIDs}/>
                    <Filter labels={datasetLabels} predictions={predictions} bind:selectedIDs={selectedIDs}/>
                </div>
                <ConfusionList 
                    jointEntropy={jointEntropy} 
                    hierarchy={hierarchy} 
                    trees={trees}
                    bind:selectedIDs={selectedIDs}
                    colorMap={colorMap}
                    numInstances={datasetLabels.length}
                />
            </div>
        </div>
    </div>
    {/if}
</main>

<style>
  #main {
    display: flex;
    flex-direction: column;
    align-items: start;
  }

  #dataset {
    display: flex;
    flex-direction: row;
    column-gap: 1em;
    align-items: center;
  }

  #content {
    display: flex;
    flex-direction: row;
    column-gap: 5em;
    overflow: hidden;
  }

  #instances {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      height: 90vh;
      width: 630px;
  }

  p {
    font-size: 12pt;
    font-weight: 600;
  }

  #right-panel {
    width: 50vw;
    display: flex;
    flex-direction: column;
    align-items: start;
    row-gap: 1em;
    height: 80vh;
  }

  #controls {
    display: flex;
    flex-direction: column;
    align-items: start;
    row-gap: 1em;
  }

</style>
