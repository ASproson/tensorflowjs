import * as tf from '@tensorflow/tfjs';
import { useCallback, useState } from 'react';


function App() {

  const [tensorData, setTensorData] = useState<tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[]>()

  const triggerModel = useCallback(() => {
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
  
    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
  
    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
  
    // Train the model using the data.
    model.fit(xs, ys).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    let res = model.predict(tf.tensor2d([10], [1, 1]))
      setTensorData(res)
  });
  }, [])

  return (
    <div className='flex justify-center mt-10'>
      <div>
      <button className="text-2xl font-bold border-2 border-black px-2 py-2" onClick={triggerModel}>
        Trigger model
      </button>
      {
        tensorData && <pre className='pt-10'>{JSON.stringify(tensorData, null, 2)}</pre>
      }
      </div>
    </div>
  );
}

export default App;
