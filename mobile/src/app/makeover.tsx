import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Linking,
  Dimensions,
} from 'react-native';
import { useLocalSearchParams, router } from 'expo-router';
import { PhotoService } from '../services/photoService';
import { BackendService } from '../services/backendService';
import { Logger } from '../utils/logger';

const { width: screenWidth } = Dimensions.get('window');

interface ProductPrice {
  retailer: string;
  price: number;
  currency: string;
  url: string;
  availability: string;
  shipping?: string;
}

interface SuggestedProduct {
  product_id: string;
  name: string;
  category: string;
  description: string;
  coordinates: number[];
  prices: ProductPrice[];
  image_url: string;
  confidence: number;
}

interface DetectedObject {
  object_type: string;
  confidence: number;
  bounding_box: number[];
  description: string;
}

interface Transformation {
  style_name: string;
  before_image_url: string;
  after_image_url: string;
  detected_objects: DetectedObject[];
  suggested_products: SuggestedProduct[];
  total_estimated_cost: number;
  savings_amount: number;
}

interface Makeover {
  makeover_id: string;
  photo_id: string;
  status: string;
  transformation?: Transformation;
  processing_time_ms: number;
  created_at: string;
}

export default function MakeoverScreen() {
  const { photoId } = useLocalSearchParams<{ photoId: string }>();
  const [makeover, setMakeover] = useState<Makeover | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedProduct, setSelectedProduct] = useState<SuggestedProduct | null>(null);
  const [showPrices, setShowPrices] = useState(false);

  useEffect(() => {
    if (photoId) {
      loadMakeover();
    }
  }, [photoId]);

  const loadMakeover = async () => {
    try {
      setLoading(true);
      const makeoverData = await PhotoService.getPhotoMakeover(photoId);
      
      if (makeoverData) {
        setMakeover(makeoverData);
      } else {
        Alert.alert('No Makeover', 'AI analysis not available for this photo.');
        router.back();
      }
    } catch (error) {
      Logger.error('Failed to load makeover', { photoId, error });
      Alert.alert('Error', 'Failed to load room makeover. Please try again.');
      router.back();
    } finally {
      setLoading(false);
    }
  };

  const handleProductPress = async (product: SuggestedProduct) => {
    setSelectedProduct(product);
    setShowPrices(true);
  };

  const handleBuyProduct = async (productPrice: ProductPrice) => {
    try {
      const supported = await Linking.canOpenURL(productPrice.url);
      if (supported) {
        await Linking.openURL(productPrice.url);
      } else {
        Alert.alert('Error', 'Cannot open product link');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to open product link');
    }
  };

  const renderProductCard = (product: SuggestedProduct) => {
    const bestPrice = product.prices.reduce((min, price) => 
      price.price < min.price ? price : min
    );

    return (
      <TouchableOpacity
        key={product.product_id}
        style={styles.productCard}
        onPress={() => handleProductPress(product)}
      >
        <Image 
          source={{ uri: product.image_url }} 
          style={styles.productImage}
          defaultSource={require('../../assets/icon.png')}
        />
        <View style={styles.productInfo}>
          <Text style={styles.productName}>{product.name}</Text>
          <Text style={styles.productCategory}>{product.category}</Text>
          <Text style={styles.productDescription}>{product.description}</Text>
          <View style={styles.priceRow}>
            <Text style={styles.bestPrice}>Best: £{bestPrice.price}</Text>
            <Text style={styles.retailer}>at {bestPrice.retailer}</Text>
          </View>
          <View style={styles.savingsRow}>
            <Text style={styles.savings}>
              Save £{(Math.max(...product.prices.map(p => p.price)) - bestPrice.price).toFixed(2)}
            </Text>
          </View>
        </View>
      </TouchableOpacity>
    );
  };

  const renderPriceComparison = () => {
    if (!selectedProduct) return null;

    return (
      <View style={styles.priceModal}>
        <View style={styles.priceModalContent}>
          <View style={styles.priceModalHeader}>
            <Text style={styles.priceModalTitle}>{selectedProduct.name}</Text>
            <TouchableOpacity
              onPress={() => setShowPrices(false)}
              style={styles.closeButton}
            >
              <Text style={styles.closeButtonText}>×</Text>
            </TouchableOpacity>
          </View>
          
          <ScrollView style={styles.pricesList}>
            {selectedProduct.prices.map((price, index) => (
              <TouchableOpacity
                key={index}
                style={styles.priceItem}
                onPress={() => handleBuyProduct(price)}
              >
                <View style={styles.priceItemLeft}>
                  <Text style={styles.retailerName}>{price.retailer}</Text>
                  <Text style={styles.availability}>{price.availability}</Text>
                  {price.shipping && (
                    <Text style={styles.shipping}>{price.shipping}</Text>
                  )}
                </View>
                <View style={styles.priceItemRight}>
                  <Text style={styles.price}>£{price.price}</Text>
                  <Text style={styles.buyButton}>Buy Now →</Text>
                </View>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Analyzing your room...</Text>
      </View>
    );
  }

  if (!makeover?.transformation) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorText}>No makeover data available</Text>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Text style={styles.backButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const { transformation } = makeover;

  return (
    <View style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.title}>Room Makeover</Text>
        </View>

        {/* Before/After Images */}
        <View style={styles.imagesSection}>
          <Text style={styles.sectionTitle}>Your {transformation.style_name} Transformation</Text>
          <Image 
            source={{ uri: transformation.before_image_url }} 
            style={styles.roomImage}
          />
          <Text style={styles.imageLabel}>Before</Text>
          <Image 
            source={{ uri: transformation.after_image_url }} 
            style={styles.roomImage}
          />
          <Text style={styles.imageLabel}>After (AI Generated)</Text>
        </View>

        {/* Cost Summary */}
        <View style={styles.costSection}>
          <Text style={styles.sectionTitle}>Shopping Summary</Text>
          <View style={styles.costRow}>
            <Text style={styles.costLabel}>Total Cost:</Text>
            <Text style={styles.totalCost}>£{transformation.total_estimated_cost}</Text>
          </View>
          <View style={styles.costRow}>
            <Text style={styles.costLabel}>You Save:</Text>
            <Text style={styles.savings}>£{transformation.savings_amount}</Text>
          </View>
        </View>

        {/* Detected Objects */}
        <View style={styles.detectedSection}>
          <Text style={styles.sectionTitle}>What We Found</Text>
          {transformation.detected_objects.map((obj, index) => (
            <View key={index} style={styles.detectedItem}>
              <Text style={styles.detectedType}>{obj.object_type}</Text>
              <Text style={styles.detectedDescription}>{obj.description}</Text>
              <Text style={styles.confidence}>
                {Math.round(obj.confidence * 100)}% confidence
              </Text>
            </View>
          ))}
        </View>

        {/* Suggested Products */}
        <View style={styles.productsSection}>
          <Text style={styles.sectionTitle}>Tap to Shop These Items</Text>
          <Text style={styles.subtitle}>
            {transformation.suggested_products.length} products • Best prices from multiple retailers
          </Text>
          
          {transformation.suggested_products.map(renderProductCard)}
        </View>
      </ScrollView>

      {/* Price Comparison Modal */}
      {showPrices && renderPriceComparison()}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
  },
  errorText: {
    fontSize: 16,
    color: '#666',
    marginBottom: 20,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e1e1e1',
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    fontSize: 16,
    color: '#007AFF',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginLeft: 16,
  },
  imagesSection: {
    padding: 16,
    backgroundColor: '#fff',
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#333',
  },
  roomImage: {
    width: screenWidth - 32,
    height: 200,
    borderRadius: 8,
    marginVertical: 8,
  },
  imageLabel: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 8,
  },
  costSection: {
    padding: 16,
    backgroundColor: '#fff',
    marginBottom: 8,
  },
  costRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  costLabel: {
    fontSize: 16,
    color: '#333',
  },
  totalCost: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  savings: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#28a745',
  },
  detectedSection: {
    padding: 16,
    backgroundColor: '#fff',
    marginBottom: 8,
  },
  detectedItem: {
    padding: 12,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    marginBottom: 8,
  },
  detectedType: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    textTransform: 'capitalize',
  },
  detectedDescription: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  confidence: {
    fontSize: 12,
    color: '#007AFF',
    marginTop: 4,
  },
  productsSection: {
    padding: 16,
    backgroundColor: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
  },
  productCard: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e1e1e1',
  },
  productImage: {
    width: 80,
    height: 80,
    borderRadius: 8,
    backgroundColor: '#ddd',
  },
  productInfo: {
    flex: 1,
    marginLeft: 12,
  },
  productName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  productCategory: {
    fontSize: 12,
    color: '#666',
    textTransform: 'uppercase',
    marginTop: 2,
  },
  productDescription: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  priceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  bestPrice: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#28a745',
  },
  retailer: {
    fontSize: 12,
    color: '#666',
    marginLeft: 8,
  },
  savingsRow: {
    marginTop: 4,
  },
  priceModal: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  priceModalContent: {
    backgroundColor: '#fff',
    borderRadius: 12,
    margin: 20,
    maxHeight: '70%',
    width: '90%',
  },
  priceModalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e1e1e1',
  },
  priceModalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    flex: 1,
  },
  closeButton: {
    padding: 8,
  },
  closeButtonText: {
    fontSize: 24,
    color: '#666',
  },
  pricesList: {
    maxHeight: 300,
  },
  priceItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  priceItemLeft: {
    flex: 1,
  },
  retailerName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  availability: {
    fontSize: 12,
    color: '#28a745',
    marginTop: 2,
  },
  shipping: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  priceItemRight: {
    alignItems: 'flex-end',
  },
  price: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  buyButton: {
    fontSize: 14,
    color: '#007AFF',
    marginTop: 4,
  },
}); 