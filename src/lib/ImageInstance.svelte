<!-- Visualization for an image instance -->
<script lang='ts'>

    export let instanceID = 0;
    export let imageLabel = '';
    export let imageURL = '';
    export let imageSize = 100;

</script>

<div class='image-instance' style='width: {imageSize}px'>
    {#if imageURL}
        <div class='metadata'>
            <span class='label'>Instance:</span>
            <span class='value'>{instanceID}</span>
        </div>
        <div class='metadata'>
            <span class='label'>Label: </span>
            <span class='value'>{imageLabel}</span>
        </div>
        <img src={imageURL} alt={`Image ${instanceID} of a ${imageLabel}`}/>
    {/if}
</div>

<style>
 .image-instance {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
 }

 .image-instance > img {
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
</style>