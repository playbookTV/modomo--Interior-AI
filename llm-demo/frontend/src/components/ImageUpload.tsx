import { useState, useCallback } from 'react';
import { Upload, X, Image as ImageIcon } from 'lucide-react';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  onImageRemove: () => void;
  selectedImage: File | null;
  disabled?: boolean;
}

export default function ImageUpload({ 
  onImageSelect, 
  onImageRemove, 
  selectedImage, 
  disabled = false 
}: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  // Handle file selection
  const handleFile = useCallback((file: File) => {
    if (file.type.startsWith('image/')) {
      onImageSelect(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  }, [onImageSelect]);

  // Handle drag events
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  // Handle drop
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (disabled) return;

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  }, [handleFile, disabled]);

  // Handle file input change
  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  // Handle image removal
  const handleRemove = useCallback(() => {
    onImageRemove();
    setPreview(null);
  }, [onImageRemove]);

  return (
    <div className="w-full">
      {!selectedImage ? (
        <div
          className={`
            border-2 border-dashed rounded-lg p-8 text-center transition-colors
            ${dragActive 
              ? 'border-blue-400 bg-blue-50' 
              : 'border-gray-300 hover:border-gray-400'
            }
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => !disabled && document.getElementById('file-input')?.click()}
        >
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            className="hidden"
            disabled={disabled}
          />
          
          <div className="flex flex-col items-center space-y-4">
            <div className={`
              w-16 h-16 rounded-full flex items-center justify-center
              ${dragActive ? 'bg-blue-100' : 'bg-gray-100'}
            `}>
              <Upload className={`
                w-8 h-8 
                ${dragActive ? 'text-blue-600' : 'text-gray-500'}
              `} />
            </div>
            
            <div>
              <p className="text-lg font-medium text-gray-900">
                Drop your image here
              </p>
              <p className="text-sm text-gray-500 mt-1">
                or click to browse files
              </p>
              <p className="text-xs text-gray-400 mt-2">
                Supports JPG, PNG, WEBP • Max 50MB
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative">
          {preview && (
            <div className="relative rounded-lg overflow-hidden bg-gray-100">
              <img
                src={preview}
                alt="Selected room"
                className="w-full h-64 object-cover"
              />
              
              <button
                onClick={handleRemove}
                className="absolute top-2 right-2 w-8 h-8 rounded-full bg-black bg-opacity-50 hover:bg-opacity-70 flex items-center justify-center transition-colors"
                disabled={disabled}
              >
                <X className="w-4 h-4 text-white" />
              </button>
              
              <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-3">
                <div className="flex items-center space-x-2">
                  <ImageIcon className="w-4 h-4" />
                  <span className="text-sm font-medium">{selectedImage.name}</span>
                  <span className="text-xs opacity-75">
                    ({(selectedImage.size / 1024 / 1024).toFixed(1)} MB)
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}