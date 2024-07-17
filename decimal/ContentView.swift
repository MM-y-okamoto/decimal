import SwiftUI
import PencilKit

// MARK: - CanvasView
struct CanvasView: UIViewRepresentable {
    @Binding var canvasView: PKCanvasView
    
    func makeUIView(context: Context) -> PKCanvasView {
        canvasView.tool = PKInkingTool(.pen, color: .black, width: 20)
        canvasView.backgroundColor = .white
        canvasView.isUserInteractionEnabled = true
        return canvasView
    }
    
    func updateUIView(_ uiView: PKCanvasView, context: Context) {}
}

// MARK: - Inference Engine
class DigitClassifier {
    private var model: VNCoreMLModel?
    
    init() {
        do {
            let modelName = "MNIST"
            
            guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
                fatalError("Failed to find \(modelName).mlmodelc. Please check the file location and project settings.")
            }
            
            let mnistModel = try MLModel(contentsOf: modelURL)
            model = try VNCoreMLModel(for: mnistModel)
        } catch {
            print("Failed to load Vision ML model: \(error)")
            fatalError("Model loading failed: \(error.localizedDescription)")
        }
    }
    
    func classify(image: UIImage, completion: @escaping (Result<VNClassificationObservation, Error>) -> Void) {
        guard let model = model,
              let ciImage = CIImage(image: image) else {
            completion(.failure(NSError(domain: "DigitClassifier", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to create CIImage"])))
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                completion(.failure(NSError(domain: "DigitClassifier", code: 1, userInfo: [NSLocalizedDescriptionKey: "No classification results"])))
                return
            }
            
            completion(.success(topResult))
        }
        
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            completion(.failure(error))
        }
    }
}

// MARK: - ContentView
struct ContentView: View {
    @State private var canvasView = PKCanvasView()
    @State private var recognizedDigit: String?
    @State private var confidence: Float?
    @State private var errorMessage: String?
    @State private var currentPath = PKStrokePath()
    private let classifier = DigitClassifier()
    
    var body: some View {
        VStack {
            CanvasView(canvasView: $canvasView)
                .frame(height: 300)
                .border(Color.gray, width: 1)
                .gesture(DragGesture(minimumDistance: 0)
                    .onChanged({ value in
                        updateStroke(at: value.location)
                    })
                        .onEnded({ _ in
                            finishStroke()
                        })
                )
            
            HStack {
                Button("Recognize") {
                    recognizeDrawing()
                }
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
                
                Button("Clear") {
                    canvasView.drawing = PKDrawing()
                    recognizedDigit = nil
                    confidence = nil
                    errorMessage = nil
                }
                .padding()
                .background(Color.red)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .padding()
            
            if let digit = recognizedDigit, let conf = confidence {
                Text("Recognized Digit: \(digit)")
                    .font(.largeTitle)
                    .padding()
                Text("Confidence: \(String(format: "%.2f%%", conf * 100))")
                    .font(.headline)
                    .padding()
            }
            
            if let error = errorMessage {
                Text("Error: \(error)")
                    .foregroundColor(.red)
                    .padding()
            }
        }
        .padding()
    }
    
    private func updateStroke(at location: CGPoint) {
        let timeInterval = ProcessInfo.processInfo.systemUptime
        let force = 1.0 as CGFloat
        let azimuth = 0.0 as CGFloat
        let altitude = CGFloat.pi / 2
        
        let strokePoint = PKStrokePoint(location: location,
                                        timeOffset: timeInterval,
                                        size: CGSize(width: 20, height: 20),
                                        opacity: 1,
                                        force: force,
                                        azimuth: azimuth,
                                        altitude: altitude)
        
        currentPath.addPoint(strokePoint)
        
        let stroke = PKStroke(ink: PKInk(.pen, color: .black), path: currentPath)
        canvasView.drawing.strokes = [stroke]
    }
    
    private func finishStroke() {
        currentPath = PKStrokePath()
    }
    
    private func recognizeDrawing() {
        let image = canvasView.drawing.image(from: canvasView.bounds, scale: UIScreen.main.scale)
        
        guard let processedImage = preprocessImage(image) else {
            errorMessage = "Failed to process the image"
            return
        }
        
        classifier.classify(image: processedImage) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let classification):
                    recognizedDigit = classification.identifier
                    confidence = classification.confidence
                    errorMessage = nil
                case .failure(let error):
                    errorMessage = error.localizedDescription
                    recognizedDigit = nil
                    confidence = nil
                }
            }
        }
    }
    
    private func preprocessImage(_ image: UIImage) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        let size = CGSize(width: 28, height: 28)
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        let context = UIGraphicsGetCurrentContext()
        context?.setFillColor(UIColor.white.cgColor)
        context?.fill(CGRect(origin: .zero, size: size))
        
        let rect = CGRect(origin: .zero, size: size)
        context?.draw(cgImage, in: rect)
        
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else { return nil }
        
        return resizedImage
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
