using System;
using System.IO;
using TensorFlow;


namespace SeeingPI
{
    class Program
    {

        static void Main(string[] args)
        {
            var graph = new TFGraph();
            var model = File.ReadAllBytes("Model/model.pb");
            var labels = File.ReadAllLines("Model/labels.txt");
            graph.Import(model);
            Console.WriteLine("Model loaded");

            var bestIdx = 0;
            float best = 0;

            using (var session = new TFSession(graph))
            {
                var tensor = ImageUtil.CreateTensorFromImageFile("Images/pic_0245.jpg");
                var runner = session.GetRunner();
                runner.AddInput(graph["Placeholder"][0], tensor).Fetch(graph["loss"][0]);
                var output = runner.Run();    
                var result = output[0];

                var probabilities = ((float[][])result.GetValue(jagged: true))[0];
                for (int i = 0; i < probabilities.Length; i++)
                {
                    if (probabilities[i] > best)
                    {
                        bestIdx = i;
                        best = probabilities[i];
                    }
                }
            }
            Console.WriteLine($"{labels[bestIdx]} ({best * 100.0}%)");
        }
    }
}
