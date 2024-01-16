<!-- Visualization for an image instance -->
<script lang='ts'>

    export let ID = 0;
    export let labels = [] as string[];
    export let instance = '';
    export let type = null as string | null;
    export let width = 100;

</script>

<div class='instance' style='width: {width}px'>
    <div class='metadata'>
        <span class='label'>Instance:</span>
        <span class='value'>{ID}</span>
    </div>
    <div class='metadata'>
        {#if labels.length > 1}
            <div class='value-list'>
                <span class='label'>Labels: </span>
                {#each labels as label}
                    <span class='value'>{label.trim()}</span>
                {/each}
            </div>
        {:else}
            <span class='label'>Label: </span>
            <span class='value'>{labels[0]}</span>
        {/if}
    </div>
    {#if instance && type == 'image'}
        <img src={instance} alt={`Image ${ID} of a ${labels}`}/>
    {/if}
</div>

<style>
 .instance {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
 }

 .instance > img {
    width: 100%;
    aspect-ratio: 1;
 }

 .metadata {
    display: flex;
    overflow: hidden;
    white-space: nowrap;
    width: 100%;
 }

 .label {
    color: grey;
    font-variant-caps: all-small-caps;
    margin-right: 3pt;
 }

 .value {
    font-family: 'Roboto-Mono', monospace;
    overflow: hidden;
    text-overflow: ellipsis;
 }

 .value-list {
    display: flex;
    flex-direction: column;
    align-items: start;
 }
</style>