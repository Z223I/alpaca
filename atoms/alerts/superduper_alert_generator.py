"""
Superduper Alert Generator - Creates Enhanced Superduper Alerts

This atom handles the creation of superduper alerts with advanced trend analysis,
momentum indicators, and enhanced messaging for Telegram notifications.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pytz


class SuperduperAlertGenerator:
    """Generates enhanced superduper alerts from filtered super alerts."""
    
    def __init__(self, superduper_alerts_dir: Path, test_mode: bool = False):
        """
        Initialize the superduper alert generator.
        
        Args:
            superduper_alerts_dir: Directory to save superduper alerts
            test_mode: Whether running in test mode
        """
        self.superduper_alerts_dir = superduper_alerts_dir
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)
    
    def _get_momentum_color_code(self, momentum: float) -> str:
        """
        Get color code for momentum text based on value.
        
        Args:
            momentum: Momentum value to color code
            
        Returns:
            HTML color code string for the momentum value
        """
        if momentum < 0.3:
            return "🔴"  # Red for < 0.3
        elif momentum < 0.5:
            return "🟡"  # Yellow for 0.3 to < 0.5
        else:
            return "🟢"  # Green for >= 0.5
    
    def create_superduper_alert(self, latest_super_alert: Dict[str, Any], 
                               trend_analysis: Dict[str, Any], 
                               trend_type: str, trend_strength: float) -> Optional[Dict[str, Any]]:
        """
        Create a superduper alert with enhanced analysis and messaging.
        
        Args:
            latest_super_alert: The triggering super alert
            trend_analysis: Detailed trend analysis data
            trend_type: Type of trend ('rising' or 'consolidating')
            trend_strength: Trend strength (0.0 to 1.0)
            
        Returns:
            Superduper alert dictionary or None if creation failed
        """
        try:
            symbol = latest_super_alert['symbol']
            
            # Create ET timestamp for when superduper alert is generated
            et_tz = pytz.timezone('US/Eastern')
            superduper_alert_time = datetime.now(et_tz)
            et_timestamp = superduper_alert_time.strftime('%Y-%m-%dT%H:%M:%S%z')
            
            # Extract key metrics from latest super alert
            signal_analysis = latest_super_alert.get('signal_analysis', {})
            current_price = signal_analysis.get('current_price', 0)
            signal_price = signal_analysis.get('signal_price', 0)
            resistance_price = signal_analysis.get('resistance_price', 0)
            penetration = signal_analysis.get('penetration_percent', 0)
            
            # Create enhanced message
            alert_message = self._create_enhanced_message(
                symbol, current_price, signal_price, resistance_price, 
                penetration, trend_type, trend_strength, trend_analysis
            )
            
            # Create superduper alert data structure
            superduper_alert = {
                "symbol": symbol,
                "timestamp": et_timestamp,
                "alert_type": "superduper_alert",
                "trigger_condition": f"{trend_type}_trend_confirmed",
                "latest_super_alert": latest_super_alert,
                "trend_analysis": {
                    "trend_type": trend_type,
                    "trend_strength": round(trend_strength, 3),
                    "timeframe_minutes": trend_analysis.get('timeframe_minutes', 45),
                    "data_points": trend_analysis.get('data_points', 0),
                    "price_change_percent": trend_analysis.get('price_change_percent', 0),
                    "price_momentum": trend_analysis.get('price_momentum', 0),
                    "penetration_change": trend_analysis.get('penetration_change', 0),
                    "avg_penetration": trend_analysis.get('avg_penetration', 0),
                    "analysis_details": trend_analysis
                },
                "enhanced_metrics": {
                    "momentum_score": self._calculate_momentum_score(trend_analysis, trend_strength),
                    "breakout_quality": self._assess_breakout_quality(latest_super_alert, trend_analysis),
                    "risk_level": self._assess_risk_level(trend_type, trend_strength, penetration),
                    "urgency_level": self._calculate_urgency_level(trend_type, trend_strength, trend_analysis)
                },
                "signal_progression": {
                    "initial_signal_price": signal_price,
                    "current_price": current_price,
                    "resistance_target": resistance_price,
                    "penetration_percent": penetration,
                    "price_above_signal": round(current_price - signal_price, 4),
                    "distance_to_resistance": round(resistance_price - current_price, 4)
                },
                "alert_message": alert_message
            }
            
            return superduper_alert
            
        except Exception as e:
            self.logger.error(f"Error creating superduper alert for {latest_super_alert.get('symbol', 'unknown')}: {e}")
            return None
    
    def _create_enhanced_message(self, symbol: str, current_price: float, 
                               signal_price: float, resistance_price: float,
                               penetration: float, trend_type: str, 
                               trend_strength: float, trend_analysis: Dict[str, Any]) -> str:
        """Create enhanced Telegram message for superduper alerts."""
        
        # Trend type emojis and descriptions
        trend_emoji = {
            'rising': '🚀📈',
            'consolidating': '🔄📊'
        }
        
        trend_desc = {
            'rising': 'STRONG UPTREND',
            'consolidating': 'CONSOLIDATING HIGH'
        }
        
        # Strength indicators
        strength_emoji = '🔥' if trend_strength > 0.7 else '⚡' if trend_strength > 0.5 else '💫'
        strength_desc = 'VERY STRONG' if trend_strength > 0.7 else 'STRONG' if trend_strength > 0.5 else 'MODERATE'
        
        # Price change and momentum
        price_change = trend_analysis.get('price_change_percent', 0)
        momentum = trend_analysis.get('price_momentum', 0)
        timeframe = trend_analysis.get('timeframe_minutes', 45)
        
        # Create the message
        message_parts = [
            f"🎯🎯 **SUPERDUPER ALERT** 🎯🎯",
            f"",
            f"{trend_emoji.get(trend_type, '📊')} **{symbol}** @ **${current_price:.4f}**",
            f"📊 **{trend_desc.get(trend_type, 'TREND')}** | {strength_emoji} **{strength_desc}**",
            f"",
            f"🎯 **Signal Performance:**",
            f"• Entry Signal: ${signal_price:.4f} ✅",
            f"• Current Price: ${current_price:.4f}",
            f"• Resistance Target: ${resistance_price:.4f}",
            f"• Penetration: **{penetration:.1f}%** into range",
            f"",
            f"📈 **Trend Analysis ({timeframe}m):**"
        ]
        
        # Add trend-specific details
        if trend_type == 'rising':
            penetration_change = trend_analysis.get('penetration_change', 0)
            momentum_color = self._get_momentum_color_code(momentum)
            message_parts.extend([
                f"• Price Movement: **+{price_change:.2f}%**",
                f"• Momentum: {momentum_color} **{momentum:.4f}%/min**",
                f"• Penetration Increase: **+{penetration_change:.1f}%**",
                f"• Pattern: **Accelerating Breakout** 🚀"
            ])
        elif trend_type == 'consolidating':
            avg_penetration = trend_analysis.get('avg_penetration', 0)
            volatility = trend_analysis.get('price_volatility', 0)
            message_parts.extend([
                f"• Price Stability: **{price_change:.2f}%** change",
                f"• Avg Penetration: **{avg_penetration:.1f}%**",
                f"• Volatility: **{volatility:.3f}** (Low)",
                f"• Pattern: **Sustained Strength** 🔄"
            ])
        
        # Add urgency and risk assessment
        urgency = self._calculate_urgency_level(trend_type, trend_strength, trend_analysis)
        risk_level = self._assess_risk_level(trend_type, trend_strength, penetration)
        
        message_parts.extend([
            f"",
            f"⚡ **Alert Level:** {urgency.upper()}",
            f"⚠️ **Risk Level:** {risk_level.upper()}",
            f"",
            f"🎯 **Action Zones:**",
            f"• Watch for continuation above ${current_price:.4f}",
            f"• Target approach to ${resistance_price:.4f}",
            f"• Monitor for volume confirmation",
            f"",
            f"⏰ **Alert Generated:** {datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M:%S ET')}"
        ])
        
        if self.test_mode:
            message_parts.insert(1, "🧪 **[TEST MODE]**")
        
        return "\\n".join(message_parts)
    
    def _calculate_momentum_score(self, trend_analysis: Dict[str, Any], trend_strength: float) -> float:
        """Calculate a momentum score from 0.0 to 1.0."""
        momentum = abs(trend_analysis.get('price_momentum', 0))
        penetration_change = abs(trend_analysis.get('penetration_change', 0))
        
        # Normalize momentum (max at 0.1% per minute)
        momentum_component = min(momentum / 0.1, 1.0)
        
        # Normalize penetration change (max at 30%)
        penetration_component = min(penetration_change / 30.0, 1.0)
        
        # Combine with trend strength
        momentum_score = (momentum_component + penetration_component + trend_strength) / 3.0
        
        return round(momentum_score, 3)
    
    def _assess_breakout_quality(self, latest_super_alert: Dict[str, Any], 
                                trend_analysis: Dict[str, Any]) -> str:
        """Assess the quality of the breakout."""
        penetration = latest_super_alert.get('signal_analysis', {}).get('penetration_percent', 0)
        volume_ratio = latest_super_alert.get('original_alert', {}).get('volume_ratio', 1.0)
        confidence = latest_super_alert.get('original_alert', {}).get('confidence_score', 0.5)
        
        # Score based on multiple factors
        quality_score = 0
        
        if penetration > 30:
            quality_score += 3
        elif penetration > 20:
            quality_score += 2
        elif penetration > 10:
            quality_score += 1
        
        if volume_ratio > 3.0:
            quality_score += 2
        elif volume_ratio > 2.0:
            quality_score += 1
        
        if confidence > 0.8:
            quality_score += 2
        elif confidence > 0.6:
            quality_score += 1
        
        # Map score to quality
        if quality_score >= 6:
            return "EXCEPTIONAL"
        elif quality_score >= 4:
            return "HIGH"
        elif quality_score >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def _assess_risk_level(self, trend_type: str, trend_strength: float, penetration: float) -> str:
        """Assess risk level based on trend characteristics."""
        if trend_type == 'rising' and trend_strength > 0.6 and penetration > 25:
            return "LOW"
        elif trend_type == 'consolidating' and trend_strength > 0.5 and penetration > 30:
            return "LOW"
        elif trend_strength > 0.4 and penetration > 15:
            return "MODERATE"
        else:
            return "HIGH"
    
    def _calculate_urgency_level(self, trend_type: str, trend_strength: float, 
                               trend_analysis: Dict[str, Any]) -> str:
        """Calculate urgency level for the alert."""
        momentum = abs(trend_analysis.get('price_momentum', 0))
        penetration_change = abs(trend_analysis.get('penetration_change', 0))
        
        urgency_score = 0
        
        # Trend strength component
        if trend_strength > 0.7:
            urgency_score += 3
        elif trend_strength > 0.5:
            urgency_score += 2
        else:
            urgency_score += 1
        
        # Momentum component
        if momentum > 0.05:
            urgency_score += 2
        elif momentum > 0.03:
            urgency_score += 1
        
        # Rising trend gets bonus urgency
        if trend_type == 'rising' and penetration_change > 15:
            urgency_score += 2
        
        # Map to urgency levels
        if urgency_score >= 6:
            return "CRITICAL"
        elif urgency_score >= 4:
            return "HIGH"
        elif urgency_score >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def save_superduper_alert(self, superduper_alert: Dict[str, Any]) -> Optional[str]:
        """
        Save superduper alert to file.
        
        Args:
            superduper_alert: Superduper alert data
            
        Returns:
            Filename if saved successfully, None otherwise
        """
        try:
            symbol = superduper_alert['symbol']
            timestamp_str = superduper_alert['timestamp']
            
            # Parse timestamp to get filename format
            et_tz = pytz.timezone('US/Eastern')
            try:
                # Handle timezone format in timestamp
                if timestamp_str.endswith(('+0000', '-0400', '-0500')):
                    clean_timestamp = timestamp_str[:-5]
                    superduper_alert_time = datetime.fromisoformat(clean_timestamp)
                else:
                    superduper_alert_time = datetime.fromisoformat(timestamp_str.replace('Z', ''))
                
                # If no timezone info, assume ET
                if superduper_alert_time.tzinfo is None:
                    superduper_alert_time = et_tz.localize(superduper_alert_time)
                else:
                    superduper_alert_time = superduper_alert_time.astimezone(et_tz)
                    
            except Exception:
                # Fallback to current time
                superduper_alert_time = datetime.now(et_tz)
            
            # Generate filename
            filename = f"superduper_alert_{symbol}_{superduper_alert_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.superduper_alerts_dir / filename
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(superduper_alert, f, indent=2)
            
            self.logger.debug(f"Superduper alert saved: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving superduper alert: {e}")
            return None
    
    def create_and_save_superduper_alert(self, latest_super_alert: Dict[str, Any], 
                                       trend_analysis: Dict[str, Any], 
                                       trend_type: str, trend_strength: float) -> Optional[str]:
        """
        Create and save a superduper alert in one operation.
        
        Args:
            latest_super_alert: The triggering super alert
            trend_analysis: Detailed trend analysis data
            trend_type: Type of trend
            trend_strength: Trend strength
            
        Returns:
            Filename if successful, None otherwise
        """
        superduper_alert = self.create_superduper_alert(
            latest_super_alert, trend_analysis, trend_type, trend_strength
        )
        
        if superduper_alert is None:
            return None
        
        filename = self.save_superduper_alert(superduper_alert)
        if filename is None:
            return None
        
        # Display superduper alert
        self._display_superduper_alert(superduper_alert, filename)
        
        return filename
    
    def _display_superduper_alert(self, superduper_alert: Dict[str, Any], filename: str) -> None:
        """Display superduper alert information."""
        try:
            symbol = superduper_alert['symbol']
            trend_analysis = superduper_alert['trend_analysis']
            enhanced_metrics = superduper_alert['enhanced_metrics']
            
            trend_type = trend_analysis['trend_type']
            trend_strength = trend_analysis['trend_strength']
            urgency_level = enhanced_metrics['urgency_level']
            
            current_price = superduper_alert['latest_super_alert']['signal_analysis']['current_price']
            penetration = superduper_alert['latest_super_alert']['signal_analysis']['penetration_percent']
            
            message = (f"🎯🎯 SUPERDUPER ALERT: {symbol} @ ${current_price:.4f}\n"
                      f"   Trend: {trend_type.upper()} | Strength: {trend_strength:.2f} | Urgency: {urgency_level}\n"
                      f"   Penetration: {penetration:.1f}% | Quality: {enhanced_metrics['breakout_quality']}\n"
                      f"   Saved: {filename}")
            
            if self.test_mode:
                print(f"[TEST MODE] {message}")
            else:
                print(message)
                
            self.logger.info(f"Superduper alert created for {symbol} - {trend_type} trend with {trend_strength:.2f} strength")
            
        except Exception as e:
            self.logger.error(f"Error displaying superduper alert: {e}")