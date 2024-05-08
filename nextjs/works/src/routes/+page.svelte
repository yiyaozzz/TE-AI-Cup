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
			body: formData
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

<body>
	<form on:submit={handleSubmit}>
		<input type="file" name="file" accept="application/pdf" />
		<button type="submit">Upload File</button>
	</form>
</body>

<style>
	body {
		font-family: 'Helvetica Neue', Arial, sans-serif;
		background-color: #f3f4f6;
		display: flex;
		justify-content: center;
		align-items: center;
		height: 100vh;
		margin: 0;
	}

	form {
		background: #fff;

		padding: 20px;
		border-radius: 8px;
		box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
		width: 300px;
	}

	input[type='file'] {
		width: 100%;
		padding: 8px;
		margin-bottom: 20px;
		border-radius: 4px;
		background-color: #fafafa;
		border: 1px solid #ddd;
		box-sizing: border-box;
		cursor: pointer;
	}

	button {
		display: block;
		width: 100%;
		padding: 10px;
		background-color: #4caf50;
		color: white;
		border: none;
		border-radius: 4px;
		cursor: pointer;
		font-size: 16px;
		transition: background-color 0.3s ease;
	}

	button:hover {
		background-color: #45a049;
	}
</style>
