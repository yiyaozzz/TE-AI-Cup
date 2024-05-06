<!-- src/routes/upload.svelte -->
<script>
// @ts-nocheck

  import { goto } from '$app/navigation';

  /**
	 * @param {{ preventDefault: () => void; target: HTMLFormElement | undefined; }} event
	 */
  async function handleSubmit(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      alert('File uploaded successfully');
      const data = await response.json();
      goto(`/validate/${data.fileId}`); // Redirect to a success page
    } else {
      console.error('Failed to upload file');
      alert('Failed to upload file');
    }
  }
</script>

<form on:submit={handleSubmit}>
  <input type="file" name="file" accept="application/pdf" />
  <button type="submit">Upload File</button>
</form>
