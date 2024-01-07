<script lang="ts">
  import {json} from "d3";
  import InstanceList from "./lib/InstanceList.svelte";
  import ProjectionPlot from "./lib/ProjectionPlotVega.svelte";
  import QueryBar from "./lib/QueryBar.svelte";
  import type { Embedding } from "./lib/types";

  let selectedIDs = [0, 1, 2,];
  let datasetName = 'cifar';

  let datasetLabels = [] as string[];
  let imageURLs = [] as string[];
  let imageLabels = [] as string[];
  let embeddings = [] as Embedding[];

  function loadLabels() {
      json(`/data/${datasetName}/labels.json`).then((jsonObject: string[]) => {
          datasetLabels = jsonObject;
      });
  }
  $: loadLabels();

  function getImageURL(instanceID: number) {
    const index = instanceID.toString().padStart(5, "0");
    return `/data/${datasetName}/images/${datasetName}_test_${index}.jpg`
  }
  $: imageURLs = selectedIDs.map(id => getImageURL(id));
  $: imageLabels = selectedIDs.map(id => datasetLabels ? datasetLabels[id] : "");


  function loadEmbeddings() {
    json(`/data/${datasetName}/embeddings.json`).then((jsonObject: Embedding[]) => {
      embeddings = jsonObject;
    })
  }
  $: loadEmbeddings();
</script>

<main>
  <div id='main'>
    <div id=instances>
      <p>Selected {selectedIDs.length} of {datasetLabels.length} instances</p>
      <InstanceList instanceIDs={selectedIDs} imageLabels={imageLabels} imageURLs={imageURLs}/>
    </div>
    <div id='embeddings'>
      <QueryBar dimensions={embeddings[0] ? embeddings[0].length : 0}/>
      <ProjectionPlot embeddings={embeddings} bind:selectedIDs={selectedIDs}/>
    </div>
  </div>
</main>

<style>
  #main {
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
  }

  p {
    font-size: 12pt;
    font-weight: 600;
  }

  #embeddings {
    width: 50vw;
    display: flex;
    flex-direction: column;
  }
</style>
