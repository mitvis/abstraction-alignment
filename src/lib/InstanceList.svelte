<!-- List of instances -->
<script lang='ts'>
    import Instance from "./Instance.svelte";
    import HierarchyPropagation from "./HierarchyPropagation.svelte";
    import IntersectionObserver from "./IntersectionObserver.svelte";
    import type { Node, Score } from "./types";
    import type { HierarchyNode } from "d3-hierarchy";

    export let instanceIDs = [] as number[];
    export let instances = [] as string[];
    export let instanceLabels = [] as string[][];
    export let instanceType = null as string | null;
    export let hierarchy = [] as Node[];
    export let scores = [] as Score[];
    export let trees = [] as HierarchyNode<Node>[];
    export let colorMap = new Map<string, string>();

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
                        <Instance ID={id} labels={instanceLabels[i]} instance={instances[i]} type={instanceType}/>
                        <HierarchyPropagation 
                            hierarchy={hierarchy}
                            scores={scores[id]}
                            root={trees[id]}
                            colorMap={colorMap}
                            on:loaded={() => {updateHeight(i)}}
                        />
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