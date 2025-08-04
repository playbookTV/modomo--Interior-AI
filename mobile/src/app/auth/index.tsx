import React from 'react';
import { View, StyleSheet, Alert } from 'react-native';
import { router } from 'expo-router';
import { useSignIn, useSignUp } from '@clerk/clerk-expo';
import { useTheme } from '@/theme/theme-provider';
import { Button, Text, Card } from '@/components/ui';

export default function AuthScreen() {
  const { theme } = useTheme();
  const { signIn, setActive } = useSignIn();
  const { signUp } = useSignUp();

  const handleGoogleSignIn = async () => {
    try {
      if (!signIn) return;

      // Start OAuth flow
      const { supportedFirstFactors } = await signIn.create({
        identifier: '',
      });

      const oauthProvider = supportedFirstFactors?.find(
        (factor) => factor.strategy === 'oauth_google'
      );

      if (oauthProvider) {
        await signIn.prepareFirstFactor({
          strategy: 'oauth_google',
          redirectUrl: 'exp://localhost:8081/auth-callback',
          actionCompleteRedirectUrl: 'exp://localhost:8081/auth-callback',
        });

        // This would open a web view or redirect to Google
        Alert.alert(
          'Sign In',
          'Google sign-in would open here in a real app. For demo purposes, simulating successful login.',
          [
            {
              text: 'Simulate Sign In',
              onPress: () => {
                // In a real app, this would be handled by the OAuth flow
                router.back();
              },
            },
          ]
        );
      }
    } catch (error) {
      console.error('Sign in error:', error);
      Alert.alert('Sign In Error', 'Failed to sign in with Google. Please try again.');
    }
  };

  const handleAppleSignIn = async () => {
    Alert.alert(
      'Apple Sign In',
      'Apple sign-in would be available here in a production app.',
      [{ text: 'OK' }]
    );
  };

  const handleEmailSignIn = async () => {
    Alert.alert(
      'Email Sign In',
      'Email sign-in form would be available here.',
      [{ text: 'OK' }]
    );
  };

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <View style={styles.content}>
        <Text variant="h1" align="center" style={styles.title}>
          Welcome to ReRoom
        </Text>
        <Text variant="h3" color="secondary" align="center" style={styles.subtitle}>
          ☁️ AI-Powered Interior Design
        </Text>
        <Text variant="body" color="muted" align="center" style={styles.description}>
          Sign in to access cloud storage, AI makeovers, and sync across all your devices
        </Text>

        <Card style={styles.authCard}>
          <Text variant="h4" style={styles.cardTitle}>Sign In</Text>
          
          <View style={styles.buttonContainer}>
            <Button
              variant="primary"
              size="lg"
              onPress={handleGoogleSignIn}
              fullWidth
              style={styles.authButton}
            >
              🌐 Continue with Google
            </Button>

            <Button
              variant="secondary"
              size="lg"
              onPress={handleAppleSignIn}
              fullWidth
              style={styles.authButton}
            >
              🍎 Continue with Apple
            </Button>

            <Button
              variant="outline"
              size="lg"
              onPress={handleEmailSignIn}
              fullWidth
              style={styles.authButton}
            >
              📧 Continue with Email
            </Button>
          </View>

          <Text variant="caption" color="secondary" align="center" style={styles.disclaimer}>
            By signing in, you agree to our Terms of Service and Privacy Policy
          </Text>
        </Card>

        <Card style={styles.featuresCard}>
          <Text variant="h4" style={styles.cardTitle}>What you get:</Text>
          
          <View style={styles.featuresList}>
            <View style={styles.featureItem}>
              <Text style={styles.featureIcon}>☁️</Text>
              <Text variant="body" style={styles.featureText}>
                Cloud storage with global CDN for fast access worldwide
              </Text>
            </View>
            
            <View style={styles.featureItem}>
              <Text style={styles.featureIcon}>🧠</Text>
              <Text variant="body" style={styles.featureText}>
                AI-powered room makeovers with real-time progress updates
              </Text>
            </View>
            
            <View style={styles.featureItem}>
              <Text style={styles.featureIcon}>🔄</Text>
              <Text variant="body" style={styles.featureText}>
                Sync photos and makeovers across all your devices
              </Text>
            </View>
            
            <View style={styles.featureItem}>
              <Text style={styles.featureIcon}>🎨</Text>
              <Text variant="body" style={styles.featureText}>
                Professional interior design suggestions with product recommendations
              </Text>
            </View>
          </View>
        </Card>

        <Button
          variant="ghost"
          size="md"
          onPress={() => router.back()}
          style={styles.skipButton}
        >
          Skip for now (local only)
        </Button>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
    maxWidth: 400,
    alignSelf: 'center',
    width: '100%',
  },
  title: {
    marginBottom: 8,
  },
  subtitle: {
    marginBottom: 12,
  },
  description: {
    marginBottom: 32,
    lineHeight: 20,
  },
  authCard: {
    padding: 24,
    marginBottom: 20,
  },
  cardTitle: {
    marginBottom: 16,
    textAlign: 'center',
  },
  buttonContainer: {
    gap: 12,
    marginBottom: 16,
  },
  authButton: {
    marginBottom: 4,
  },
  disclaimer: {
    lineHeight: 16,
  },
  featuresCard: {
    padding: 20,
    marginBottom: 20,
  },
  featuresList: {
    gap: 12,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
  },
  featureIcon: {
    fontSize: 20,
    width: 24,
    textAlign: 'center',
  },
  featureText: {
    flex: 1,
    lineHeight: 20,
  },
  skipButton: {
    alignSelf: 'center',
  },
});