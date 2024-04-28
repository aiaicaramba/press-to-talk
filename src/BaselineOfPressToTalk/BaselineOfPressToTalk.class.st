Class {
	#name : #BaselineOfPressToTalk,
	#superclass : #BaselineOf,
	#category : #BaselineOfPressToTalk
}

{ #category : #baseline }
BaselineOfPressToTalk >> baseline: spec [
	<baseline>
		^ spec for: #common do: [
			spec package: 'PressToTalk'
		]
]
