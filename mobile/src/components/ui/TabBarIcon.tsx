/**
 * Tab Bar Icon Component
 * Simple icon component for tab navigation
 */

import React from 'react';
import { Text, StyleSheet } from 'react-native';
import { useTheme } from '@/theme/theme-provider';

interface TabBarIconProps {
  name: string;
  color: string;
  size?: number;
}

const iconMap: Record<string, string> = {
  'home': '🏠',
  'home-outline': '🏡',
  'camera': '📸',
  'camera-outline': '📷',
  'bookmark': '🔖',
  'bookmark-outline': '📋',
  'person': '👤',
  'person-outline': '👥',
};

export function TabBarIcon({ name, color, size = 24 }: TabBarIconProps) {
  const { theme } = useTheme();
  
  return (
    <Text style={[
      styles.icon,
      {
        color,
        fontSize: size,
      }
    ]}>
      {iconMap[name] || '📱'}
    </Text>
  );
}

const styles = StyleSheet.create({
  icon: {
    textAlign: 'center',
  },
});