import { error } from '@sveltejs/kit';
import { promises as fs } from 'fs';
import path from 'path';

const PROCESS_DIR = path.resolve('processing');

export async function GET({ request, url }) {
	if (!checkIfFileExists(`${url.searchParams.get('id')}.xlsx`)) {
		error(404, 'File not found');
	}

	try {
		const imageBuffer = await fs.readFile(`processing/${url.searchParams.get('id')}.xlsx`);

		return new Response(imageBuffer, {
			status: 200,
			headers: {
				'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
			}
		});
	} catch (error) {
		console.error('Error reading image:', error);
		return new Response('Image not found', {
			status: 404,
			headers: {
				'Content-Type': 'text/plain'
			}
		});
	}
}

/**
 * @param {string} fileName
 */
function checkIfFileExists(fileName) {
	// Check if the file exists in uploads folder
	console.log(path.join(PROCESS_DIR, fileName));
	return fs
		.stat(path.join(PROCESS_DIR, fileName))
		.then(() => true)
		.catch(() => false);
}