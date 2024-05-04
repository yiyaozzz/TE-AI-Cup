const express = require('express');
const multer = require('multer');
const path = require('path');
const { exec } = require('child_process');

const app = express();
const port = 3002;

// Configure multer for file storage
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')  // Make sure this directory exists
    },
    filename: function (req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname))
    }
});

const upload = multer({ storage: storage });

// Serve HTML form at root
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));  // Ensure the 'index.html' is in the same directory as this file
});

// Handle file uploads and process with Python
app.post('/upload', upload.single('fileUpload'), (req, res) => {
    if (req.file) {
        // Call a Python script to process the uploaded file
        exec(`cd main && python3 main.py nextjs/example/"${req.file.path}"`, (err, stdout, stderr) => {
            if (err) {
                console.error(err);
                return res.status(500).send(`Error during file processing: ${err.message}`);
            }
            if (stderr) {
                console.error(stderr);
                return res.status(500).send(`Error during file processing: ${stderr}`);
            }

            // Assuming the Python script outputs the path to the processed PDF
            console.log(`Processed file path: ${stdout}`);
            res.send(`File has been processed successfully. Download link: ${stdout.trim()}`);
        });
    } else {
        res.status(400).send('No file uploaded.');
    }
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
