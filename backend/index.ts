import express, { Request, Response } from 'express';
import * as tf from '@tensorflow/tfjs-node';
import multer from 'multer';
import cors from 'cors';
import fs from 'fs';

const app = express();
app.use(cors());
const upload = multer({ dest: 'uploads/' });
const modelPath = './models/model.json';

app.post('/prediction', upload.single('image'), async (req: Request, res: Response) => {

  try {
    const imagePath = req.file?.path;
    const buffer = fs.readFileSync(imagePath ?? '');

    // Load the image buffer into a TensorFlow.js tensor
    const tensor = tf.node.decodeImage(buffer, 3).resizeBilinear([256, 256]).expandDims(0);
    
    const model = await tf.loadLayersModel('file://' + modelPath);
    const prediction = model.predict(tensor);

    // Get the predicted class index and probability
    //@ts-ignore
    const predictedClassIndex = tf.argMax(prediction).dataSync()[0];
    //@ts-ignore
    const predictedProbability = prediction.dataSync()[predictedClassIndex];

    // Return the prediction as a JSON object in the response
    res.json({ 
      classIndex: predictedClassIndex,
      probability: predictedProbability
    });
  }catch (error) {
    // Handle any errors that may occur
    console.error(error);
    res.status(500).json({ error: 'Server error' });
  }
});

app.listen(4000, () => {
  console.log('Server listening on port 4000');
});
