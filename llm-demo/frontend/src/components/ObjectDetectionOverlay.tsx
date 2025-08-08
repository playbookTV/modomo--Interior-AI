import { useState } from 'react';
import type { DetectedObject, SuggestedProduct } from '@/types/api';
import { Info, ShoppingBag, ExternalLink } from 'lucide-react';

interface ObjectDetectionOverlayProps {
  imageUrl: string;
  detectedObjects: DetectedObject[];
  suggestedProducts: SuggestedProduct[];
  imageWidth: number;
  imageHeight: number;
}

export default function ObjectDetectionOverlay({
  imageUrl,
  detectedObjects,
  suggestedProducts,
  imageWidth,
  imageHeight,
}: ObjectDetectionOverlayProps) {
  const [selectedObject, setSelectedObject] = useState<DetectedObject | null>(null);
  const [selectedProduct, setSelectedProduct] = useState<SuggestedProduct | null>(null);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [showProductMarkers, setShowProductMarkers] = useState(true);

  const handleObjectClick = (obj: DetectedObject) => {
    setSelectedObject(obj);
    setSelectedProduct(null);
  };

  const handleProductClick = (product: SuggestedProduct) => {
    setSelectedProduct(product);
    setSelectedObject(null);
  };

  const closeModals = () => {
    setSelectedObject(null);
    setSelectedProduct(null);
  };

  return (
    <div className="relative">
      {/* Controls */}
      <div className="absolute top-4 left-4 z-20 bg-white rounded-lg shadow-lg p-3 space-y-2">
        <label className="flex items-center space-x-2 text-sm">
          <input
            type="checkbox"
            checked={showBoundingBoxes}
            onChange={(e) => setShowBoundingBoxes(e.target.checked)}
            className="rounded"
          />
          <span>Show Objects</span>
        </label>
        <label className="flex items-center space-x-2 text-sm">
          <input
            type="checkbox"
            checked={showProductMarkers}
            onChange={(e) => setShowProductMarkers(e.target.checked)}
            className="rounded"
          />
          <span>Show Products</span>
        </label>
      </div>

      {/* Main image with overlays */}
      <div className="relative">
        <img
          src={imageUrl}
          alt="Room analysis"
          className="w-full h-auto rounded-lg"
          style={{ maxWidth: imageWidth, maxHeight: imageHeight }}
        />

        {/* Detected Objects Overlay */}
        {showBoundingBoxes && detectedObjects.map((obj, index) => {
          const [x, y, width, height] = obj.bounding_box;
          const scaleX = imageWidth / 1000; // Assuming normalized coordinates
          const scaleY = imageHeight / 1000;

          return (
            <div
              key={`object-${index}`}
              className="absolute border-2 border-blue-500 bg-blue-500 bg-opacity-10 cursor-pointer hover:bg-opacity-20 transition-colors"
              style={{
                left: x * scaleX,
                top: y * scaleY,
                width: width * scaleX,
                height: height * scaleY,
              }}
              onClick={() => handleObjectClick(obj)}
            >
              <div className="absolute -top-6 left-0 bg-blue-500 text-white text-xs px-2 py-1 rounded whitespace-nowrap">
                {obj.object_type} ({Math.round(obj.confidence * 100)}%)
              </div>
            </div>
          );
        })}

        {/* Suggested Products Overlay */}
        {showProductMarkers && suggestedProducts.map((product, index) => {
          const [x, y] = product.coordinates;
          const scaleX = imageWidth / 1000;
          const scaleY = imageHeight / 1000;

          return (
            <div
              key={`product-${index}`}
              className="absolute w-6 h-6 bg-green-500 rounded-full flex items-center justify-center cursor-pointer hover:bg-green-600 transition-colors shadow-lg"
              style={{
                left: x * scaleX - 12,
                top: y * scaleY - 12,
              }}
              onClick={() => handleProductClick(product)}
            >
              <ShoppingBag className="w-3 h-3 text-white" />
            </div>
          );
        })}
      </div>

      {/* Object Details Modal */}
      {selectedObject && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center space-x-2">
                <Info className="w-5 h-5 text-blue-500" />
                <span>Object Details</span>
              </h3>
              <button
                onClick={closeModals}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium text-gray-700">Type:</label>
                <p className="text-gray-900">{selectedObject.object_type}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-gray-700">Confidence:</label>
                <p className="text-gray-900">{Math.round(selectedObject.confidence * 100)}%</p>
              </div>
              <div>
                <label className="text-sm font-medium text-gray-700">Description:</label>
                <p className="text-gray-900">{selectedObject.description}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-gray-700">Bounding Box:</label>
                <p className="text-gray-600 text-sm">
                  [{selectedObject.bounding_box.join(', ')}]
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Product Details Modal */}
      {selectedProduct && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full max-h-screen overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center space-x-2">
                <ShoppingBag className="w-5 h-5 text-green-500" />
                <span>Product Details</span>
              </h3>
              <button
                onClick={closeModals}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-900 mb-1">{selectedProduct.name}</h4>
                <p className="text-sm text-gray-600">{selectedProduct.description}</p>
                <span className="inline-block mt-1 px-2 py-1 bg-gray-100 text-xs text-gray-700 rounded">
                  {selectedProduct.category}
                </span>
              </div>
              
              <div>
                <label className="text-sm font-medium text-gray-700">Confidence:</label>
                <p className="text-gray-900">{Math.round(selectedProduct.confidence * 100)}%</p>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">Prices:</label>
                <div className="space-y-2">
                  {selectedProduct.prices.slice(0, 3).map((price, index) => (
                    <div key={index} className="flex items-center justify-between p-2 border rounded">
                      <div>
                        <p className="font-medium">{price.retailer}</p>
                        <p className="text-sm text-gray-600">{price.availability}</p>
                        {price.shipping && (
                          <p className="text-xs text-gray-500">{price.shipping}</p>
                        )}
                      </div>
                      <div className="text-right">
                        <p className="font-bold text-green-600">
                          {price.currency} {price.price.toFixed(2)}
                        </p>
                        <a
                          href={price.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center text-xs text-blue-600 hover:text-blue-800"
                        >
                          View <ExternalLink className="w-3 h-3 ml-1" />
                        </a>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Backdrop click handler */}
      {(selectedObject || selectedProduct) && (
        <div
          className="fixed inset-0 z-40"
          onClick={closeModals}
        />
      )}
    </div>
  );
}