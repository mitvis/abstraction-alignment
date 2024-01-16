<!-- List of instances -->
<script lang='ts'>
    import Instance from "./Instance.svelte";
    import HierarchyPropagation from "./HierarchyPropagation.svelte";
    import IntersectionObserver from "./IntersectionObserver.svelte";

    export let instanceIDs = [] as number[];
    export let instances = [] as string[];
    export let instanceLabels = [] as string[][];
    export let instanceType = null as string | null;

    // Update the height of the instance container when the hierarchy loads
    let instanceContainers: any[] = [];

    function updateHeight(i: number) {
        instanceContainers[i].style.height = 'auto';
        instanceContainers[i].style.minHeight = 'auto';
    }

</script>

<div id='instance-list'>
    {#each instanceIDs as id, i}
        <div class='instance-container' bind:this={instanceContainers[i]}>
            <IntersectionObserver once={true} let:intersecting={intersecting}>
                {#if intersecting}
                    <div class='instance'>
                        <HierarchyPropagation instanceID={id} on:loaded={() => updateHeight(i)}/>
                        <Instance ID={id} labels={instanceLabels[i]} instance={instances[i]} type={instanceType}/>
                    </div>
                {/if}
            </IntersectionObserver>
        </div>
    {/each}
</div>

<style>
    .instance-container {
        min-height: 20%;
    }

    .instance {
        display: flex;
        flex-direction: row;
        column-gap: 1em;
        width: 100%;
        height: 100%;
    }

    #instance-list {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      row-gap: 2em;
      height: 100%;
      width: 100%;
      overflow-y: scroll;
  }
</style>