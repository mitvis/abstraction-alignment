<!-- List of instances -->
<script lang='ts'>
    import ImageInstance from "./ImageInstance.svelte";
    import HierarchyPropagation from "./HierarchyPropagation.svelte";

    export let instanceIDs = [] as number[];
    export let imageURLs = [] as string[];
    export let imageLabels = [] as string[];
</script>

<div id='instance-list'>
    {#each instanceIDs as id, i}
        <div class='instance'>
            <ImageInstance instanceID={id} imageLabel={imageLabels[i]} imageURL={imageURLs[i]}/>
            <HierarchyPropagation instanceID={id}/>
        </div>
    {/each}
</div>

<style>
    .instance {
        display: flex;
        flex-direction: row;
        column-gap: 1em;
    }

    #instance-list {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      row-gap: 2em;
      height: 100%;
      overflow-y: scroll;
  }
</style>