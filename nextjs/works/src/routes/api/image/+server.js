// src/routes/api/image.js
import { error } from '@sveltejs/kit';
import { promises as fs } from 'fs';
import path from 'path';

const PROCESS_DIR = path.resolve('processing');

export async function GET({ request, url }) {
	// console.log(url.searchParams.get('name'));

	if (
		!checkIfFileExists(`${url.searchParams.get('folderId')}.json`) ||
		!url.searchParams.get('page')
	) {
		error(404, 'File not found');
		return;
	}

	try {
		// let name = await getFirstNameInDirectory(
		// 	`uploads/${url.searchParams.get('folderId')}_pages/page_${url.searchParams.get('page')}/row_${url.searchParams.get('row')}/column_${url.searchParams.get('col')}/`
		// );
		const imageBuffer = await fs.readFile(
			`uploads/${url.searchParams.get('folderId')}_pages/page_${url.searchParams.get('page')}.png`
		);

		return new Response(imageBuffer, {
			status: 200,
			headers: {
				'Content-Type': 'image/png'
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

/**
 * @param {import("fs").PathLike} dirPath
 */
async function getFirstNameInDirectory(dirPath) {
	try {
		const files = await fs.readdir(dirPath);
		if (files.length === 0) {
			return 'No files found in the directory.';
		}
		return files[0]; // Returns the name of the first file
	} catch (error) {
		console.error('Error accessing the directory:', error);
		return 'Failed to read the directory.';
	}
}
