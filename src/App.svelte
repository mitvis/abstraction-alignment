<script lang="ts">
  import {json} from "d3";
  import InstanceList from "./lib/InstanceList.svelte";
  import Filter from "./lib/Filter.svelte";
  import ProjectionPlot from "./lib/ProjectionPlot.svelte";
  import QueryBar from "./lib/QueryBar.svelte";
  import type { Embedding } from "./lib/types";
  

  const datasets = ['cifar', 'mimic'];
  let datasetName = 'cifar';
  let selectedIDs = [] as number[];

  let datasetLabels = [] as string[];
  let imageURLs = [] as string[];
  let imageLabels = [] as string[];
  let embeddings = [] as Embedding[];

  function loadLabels(datasetName: string) {
      json(`/data/${datasetName}/labels.json`).then((jsonObject: string[]) => {
          datasetLabels = jsonObject;
      });
  }
  $: loadLabels(datasetName);

  function getImageURL(instanceID: number, datasetName: string) {
    const index = instanceID.toString().padStart(5, "0");
    return `/data/${datasetName}/images/${datasetName}_test_${index}.jpg`
  }
  $: imageURLs = selectedIDs.map(id => getImageURL(id, datasetName));
  $: imageLabels = selectedIDs.map(id => datasetLabels ? datasetLabels[id] : "");


  function loadEmbeddings(datasetName: string) {
    json(`/data/${datasetName}/embeddings.json`).then((jsonObject: Embedding[]) => {
      embeddings = jsonObject.map(embedding => embedding.map(value => parseFloat(value.toFixed(2))));
      selectedIDs = Array.from({length: embeddings.length}, (_, i) => i);
    })
  }
  $: loadEmbeddings(datasetName);
</script>

<main>
  <div id='main'>
    <div id='dataset'>
      <p>Dataset: </p>
      {#each datasets as dataset}
        <button class:active={datasetName == dataset} on:click={() => datasetName = dataset}>
          {dataset}
        </button>
      {/each}
    </div>
    <div id='content'>
      <div id=instances>
        <p>Selected {selectedIDs.length} of {datasetLabels.length} instances</p>
        <InstanceList instanceIDs={selectedIDs} imageLabels={imageLabels} imageURLs={imageURLs}/>
      </div>
      <div id='embeddings'>
        <QueryBar embeddings={embeddings} bind:selectedIDs={selectedIDs}/>
        <Filter labels={datasetLabels} bind:selectedIDs={selectedIDs}/>
        <ProjectionPlot embeddings={embeddings} bind:selectedIDs={selectedIDs}/>
      </div>
    </div>
  </div>
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

  #embeddings {
    width: 50vw;
    display: flex;
    flex-direction: column;
    align-items: start;
    row-gap: 1em;
  }

  .active {
        background-color: #ddd;
    }
</style>
