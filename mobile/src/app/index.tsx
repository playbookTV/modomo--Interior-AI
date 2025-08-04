import { View, StyleSheet } from 'react-native';
import { router } from 'expo-router';
import { useAuth } from '@clerk/clerk-expo';
import { useTheme } from '@/theme/theme-provider';
import { Button, Text, Card } from '@/components/ui';

export default function HomeScreen() {
  const { theme } = useTheme();
  const auth = useAuth();
  
  const handleStartCapture = () => {
    router.push('/camera');
  };

  const handleViewGallery = () => {
    router.push('/gallery');
  };

  const handleSignIn = () => {
    router.push('/auth');
  };

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Text variant="h1" align="center" style={styles.title}>
        Welcome to ReRoom! ☁️
      </Text>
      <Text variant="h4" color="secondary" align="center" style={styles.subtitle}>
        AI-Powered Interior Design in the Cloud
      </Text>
      <Text variant="body" color="muted" align="center" style={styles.tagline}>
        Capture. Transform. Share.
      </Text>
      
      {/* Authentication Status */}
      {auth.isSignedIn ? (
        <Card style={styles.statusCard}>
          <Text variant="h4" style={styles.statusTitle}>✅ Signed In</Text>
          <Text variant="body" color="secondary" align="center">
            Cloud storage active • AI processing enabled • Global sync ready
          </Text>
        </Card>
      ) : (
        <Card style={styles.statusCard}>
          <Text variant="h4" style={styles.statusTitle}>🔐 Sign In for Full Features</Text>
          <Text variant="body" color="secondary" align="center" style={styles.statusText}>
            Access cloud storage, AI makeovers, and sync across devices
          </Text>
          <Button 
            variant="primary" 
            size="md" 
            onPress={handleSignIn}
            style={styles.signInButton}
          >
            Sign In Now
          </Button>
        </Card>
      )}
      
      <View style={styles.buttonContainer}>
        <Button 
          variant="primary" 
          size="lg" 
          fullWidth
          onPress={handleStartCapture}
          style={styles.primaryButton}
        >
          📸 Capture Your Room
        </Button>
        
        <Button 
          variant="secondary" 
          size="lg" 
          fullWidth
          onPress={handleViewGallery}
          style={styles.secondaryButton}
        >
          🖼️ View Cloud Gallery
        </Button>
        
        <Text variant="body" color="secondary" align="center" style={styles.description}>
          {auth.isSignedIn 
            ? 'Take a photo and get AI makeovers stored in the cloud with global CDN delivery'
            : 'Take photos locally or sign in for cloud storage and AI-powered transformations'
          }
        </Text>
      </View>

      {/* Feature Highlights */}
      <View style={styles.featuresContainer}>
        <Text variant="h4" align="center" style={styles.featuresTitle}>
          New Cloud Features:
        </Text>
        <View style={styles.featuresList}>
          <Text variant="body" color="secondary" align="center" style={styles.featureItem}>
            ☁️ Global cloud storage with CDN
          </Text>
          <Text variant="body" color="secondary" align="center" style={styles.featureItem}>
            🧠 Real-time AI processing updates
          </Text>
          <Text variant="body" color="secondary" align="center" style={styles.featureItem}>
            🔄 Sync across all your devices
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    marginBottom: 10,
  },
  subtitle: {
    marginBottom: 5,
  },
  tagline: {
    fontStyle: 'italic',
    marginBottom: 40,
  },
  buttonContainer: {
    alignItems: 'center',
    width: '100%',
    maxWidth: 320,
  },
  primaryButton: {
    marginBottom: 16,
  },
  secondaryButton: {
    marginBottom: 24,
  },
  description: {
    maxWidth: 320,
  },
  statusCard: {
    marginHorizontal: 20,
    marginVertical: 16,
    padding: 16,
    alignItems: 'center',
  },
  statusTitle: {
    marginBottom: 8,
    textAlign: 'center',
  },
  statusText: {
    marginBottom: 12,
    textAlign: 'center',
  },
  signInButton: {
    marginTop: 4,
  },
  featuresContainer: {
    marginTop: 20,
    paddingHorizontal: 20,
  },
  featuresTitle: {
    marginBottom: 12,
  },
  featuresList: {
    gap: 4,
  },
  featureItem: {
    lineHeight: 20,
  },
}); 